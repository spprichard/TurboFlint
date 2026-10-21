#!/usr/bin/env python3
"""Layer-by-layer FP32 reference forward pass for Recursive Flint (40B-A2.5B).

Reads the MLX affine-int4 checkpoint straight out of the Ollama blob store, dequantizes only the
tensors each step needs (routed experts are sliced per selected expert), and never holds the whole
model in memory. It is deliberately slow and simple: it is the numerical oracle the Metal engine is
checked against.

The math follows recursive-oss/rec engine/src/rec_mlx/model.py:
  - pre-norm decoder; layers 0-1 dense SiLU MLP, 2-48 sparse MoE + always-on shared expert
  - per-head QK RMSNorm; non-traditional (half-split) RoPE only on sliding layers
  - per-head learnable sink logit appended to the softmax, then dropped
  - sliding window 513 (current token + 512 previous); full layers use no positional encoding
  - router: sigmoid(x @ W.T), top-6, weights renormalised; routed sum + shared expert

Usage:
  python3 -I Scripts/reference/flint_reference.py --prompt "The capital of France is" --max-new 8
  python3 -I Scripts/reference/flint_reference.py --prompt-ids 785,6722 --out ref.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEFAULT_MANIFEST = Path.home() / ".ollama/models/manifests/registry.ollama.ai/library/flint/40b-a2b-qat-int4"
GROUP = 64


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)


class BlobStore:
    """Maps checkpoint tensor names to read-only memmaps inside Ollama's per-tensor safetensors blobs."""

    def __init__(self, manifest_path: Path):
        manifest = json.loads(manifest_path.read_text())
        blob_dir = manifest_path.parents[4] / "blobs"
        self.tensors: dict[str, tuple[Path, int, str, list[int]]] = {}
        self.json_blobs: dict[str, Path] = {}
        for layer in manifest["layers"]:
            path = blob_dir / layer["digest"].replace(":", "-")
            media = layer["mediaType"]
            if media.endswith(".tensor"):
                with open(path, "rb") as f:
                    (n,) = struct.unpack("<Q", f.read(8))
                    header = json.loads(f.read(n))
                for name, info in header.items():
                    if name == "__metadata__":
                        continue
                    start = 8 + n + info["data_offsets"][0]
                    self.tensors[name] = (path, start, info["dtype"], info["shape"])
            elif media.endswith(".json") and "name" in layer:
                self.json_blobs[layer["name"]] = path
        self._maps: dict[Path, np.memmap] = {}

    def raw(self, name: str) -> np.ndarray:
        path, start, dtype, shape = self.tensors[name]
        np_dtype = {"U32": np.uint32, "BF16": np.uint16, "F32": np.float32, "U8": np.uint8}[dtype]
        mm = self._maps.get(path)
        if mm is None:
            mm = self._maps[path] = np.memmap(path, dtype=np.uint8, mode="r")
        count = int(np.prod(shape))
        return mm[start : start + count * np.dtype(np_dtype).itemsize].view(np_dtype).reshape(shape)

    def f32(self, name: str) -> torch.Tensor:
        raw = self.raw(name)
        if raw.dtype == np.uint16:
            raw = bf16_to_f32(raw)
        return torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32))

    def json(self, name: str) -> dict:
        return json.loads(self.json_blobs[name].read_text())


def dequant(w: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> torch.Tensor:
    """MLX affine dequant: value = scale * q + bias per 64-wide group; q packed LSB-first in u32."""
    groups = scale.shape[-1]
    in_features = groups * GROUP
    bits = w.shape[-1] * 32 // in_features
    per_word = 32 // bits
    shifts = (np.arange(per_word, dtype=np.uint32) * bits).astype(np.uint32)
    q = (w[..., None] >> shifts) & np.uint32((1 << bits) - 1)
    q = q.reshape(*w.shape[:-1], groups, GROUP).astype(np.float32)
    s = bf16_to_f32(scale)[..., None]
    b = bf16_to_f32(bias)[..., None]
    return torch.from_numpy((q * s + b).reshape(*w.shape[:-1], in_features))


class Flint:
    def __init__(self, store: BlobStore):
        self.s = store
        c = store.json("config.json")
        self.c = c
        self.hidden = c["hidden_size"]
        self.n_heads = c["num_attention_heads"]
        self.n_kv = c["num_key_value_heads"]
        self.head_dim = c["head_dim"]
        self.top_k = c["num_experts_per_tok"]
        self.eps = c["rms_norm_eps"]
        self.window = c["sliding_window"]
        self.no_rope = set(c["no_rope_layer_indices"])
        self.dense = set(c["mlp_only_layers"])
        self.layer_types = c["layer_types"]
        self.n_layers = c["num_hidden_layers"]
        self.theta = c["rope_theta"]
        inv = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float64) / self.head_dim))
        self.inv_freq = inv.float()
        # Per-layer KV cache, full history in FP32 (post-norm, post-RoPE keys).
        self.k_cache: list[torch.Tensor | None] = [None] * self.n_layers
        self.v_cache: list[torch.Tensor | None] = [None] * self.n_layers
        self.pos = 0
        self.stats = {"expert_loads": 0}

    # -- weights ---------------------------------------------------------------------------------
    def qweight(self, prefix: str, expert: int | None = None) -> torch.Tensor:
        w, s, b = (self.s.raw(prefix), self.s.raw(prefix + ".scale"), self.s.raw(prefix + ".bias"))
        if expert is not None:
            w, s, b = w[expert], s[expert], b[expert]
        return dequant(np.asarray(w), np.asarray(s), np.asarray(b))

    def linear(self, x: torch.Tensor, prefix: str, expert: int | None = None) -> torch.Tensor:
        return x @ self.qweight(prefix, expert).T

    def rms(self, x: torch.Tensor, weight_name: str) -> torch.Tensor:
        w = self.s.f32(weight_name)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def rms_heads(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def embed(self, ids: list[int]) -> torch.Tensor:
        name = "model.embed_tokens.weight"
        idx = np.asarray(ids)
        w = np.asarray(self.s.raw(name))[idx]
        s = np.asarray(self.s.raw(name + ".scale"))[idx]
        b = np.asarray(self.s.raw(name + ".bias"))[idx]
        return dequant(w, s, b)

    # -- blocks ----------------------------------------------------------------------------------
    def rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: [L, heads, D]; non-traditional RoPE rotates (first half, second half) pairs.
        ang = positions[:, None].float() * self.inv_freq[None, :]  # [L, D/2]
        cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
        half = self.head_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def attention(self, i: int, x: torch.Tensor) -> torch.Tensor:
        p = f"model.layers.{i}.self_attn."
        L = x.shape[0]
        q = self.linear(x, p + "q_proj.weight").reshape(L, self.n_heads, self.head_dim)
        k = self.linear(x, p + "k_proj.weight").reshape(L, self.n_kv, self.head_dim)
        v = self.linear(x, p + "v_proj.weight").reshape(L, self.n_kv, self.head_dim)
        q = self.rms_heads(q, self.s.f32(p + "q_norm.weight"))
        k = self.rms_heads(k, self.s.f32(p + "k_norm.weight"))
        positions = torch.arange(self.pos, self.pos + L)
        if i not in self.no_rope:
            q, k = self.rope(q, positions), self.rope(k, positions)
        if self.k_cache[i] is None:
            self.k_cache[i], self.v_cache[i] = k, v
        else:
            self.k_cache[i] = torch.cat([self.k_cache[i], k])
            self.v_cache[i] = torch.cat([self.v_cache[i], v])
        K, V = self.k_cache[i], self.v_cache[i]  # [T, n_kv, D]
        T = K.shape[0]
        rep = self.n_heads // self.n_kv
        Kh = K.permute(1, 0, 2).repeat_interleave(rep, dim=0)  # [H, T, D]
        Vh = V.permute(1, 0, 2).repeat_interleave(rep, dim=0)
        scores = torch.einsum("lhd,htd->hlt", q, Kh) / math.sqrt(self.head_dim)  # [H, L, T]
        qi = positions[:, None]
        kj = torch.arange(T)[None, :]
        allowed = kj <= qi
        if self.layer_types[i] == "sliding_attention":
            allowed &= qi < kj + self.window
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        sinks = self.s.f32(p + "softmax_offset").reshape(self.n_heads, 1, 1).expand(-1, L, 1)
        probs = torch.softmax(torch.cat([scores, sinks], dim=-1), dim=-1)[..., :T]
        out = torch.einsum("hlt,htd->lhd", probs, Vh).reshape(L, self.n_heads * self.head_dim)
        return self.linear(out, p + "o_proj.weight")

    def mlp(self, x: torch.Tensor, prefix: str, expert: int | None = None) -> torch.Tensor:
        g = self.linear(x, prefix + "gate_proj.weight", expert)
        u = self.linear(x, prefix + "up_proj.weight", expert)
        return self.linear(torch.nn.functional.silu(g) * u, prefix + "down_proj.weight", expert)

    def moe(self, i: int, x: torch.Tensor, trace: dict | None) -> torch.Tensor:
        p = f"model.layers.{i}.mlp."
        router = self.s.f32(p + "gate.weight")  # [256, hidden]
        scores = torch.sigmoid(x @ router.T)  # [L, 256]
        top = torch.topk(scores, self.top_k, dim=-1)
        weights = top.values / (top.values.sum(-1, keepdim=True) + 1e-20)
        out = torch.zeros_like(x)
        for e in sorted(set(top.indices.flatten().tolist())):
            rows = (top.indices == e).nonzero()
            r, slot = rows[:, 0], rows[:, 1]
            y = self.mlp(x[r], p + "experts.", expert=e)
            out.index_add_(0, r, y * weights[r, slot][:, None])
            self.stats["expert_loads"] += 1
        if trace is not None:
            trace.setdefault("routing", []).append(
                {"layer": i, "experts": top.indices[-1].tolist(), "weights": weights[-1].tolist()}
            )
        return out + self.mlp(x, p + "shared_expert.")

    def forward(self, ids: list[int], trace: dict | None = None) -> torch.Tensor:
        """Runs `ids` as the next tokens; returns logits for the last one."""
        h = self.embed(ids)
        hidden_trace = []
        for i in range(self.n_layers):
            p = f"model.layers.{i}."
            h = h + self.attention(i, self.rms(h, p + "input_layernorm.weight"))
            n = self.rms(h, p + "post_attention_layernorm.weight")
            h = h + (self.mlp(n, p + "mlp.") if i in self.dense else self.moe(i, n, trace))
            hidden_trace.append(h[-1].clone())
        self.pos += len(ids)
        last = self.rms(h[-1:], "model.norm.weight")
        logits = self.lm_head(last)[0]
        if trace is not None:
            trace["hidden_last_row"] = hidden_trace
        return logits

    def lm_head(self, x: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
        name = "lm_head.weight"
        w, s, b = self.s.raw(name), self.s.raw(name + ".scale"), self.s.raw(name + ".bias")
        parts = []
        for start in range(0, w.shape[0], chunk):
            sl = slice(start, start + chunk)
            parts.append(x @ dequant(np.asarray(w[sl]), np.asarray(s[sl]), np.asarray(b[sl])).T)
        return torch.cat(parts, dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="raw text (no chat template, no BOS)")
    src.add_argument("--chat", help="user message rendered as <|user|>...<|assistant|>")
    src.add_argument("--prompt-ids", help="comma-separated token ids")
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--top", type=int, default=5, help="top-k logits recorded per step")
    ap.add_argument("--dump-hidden", action="store_true", help="record per-layer hidden state of the last prompt row")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    store = BlobStore(args.manifest)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(store.json_blobs["tokenizer.json"]))
    if args.prompt_ids:
        ids = [int(t) for t in args.prompt_ids.split(",")]
        text = None
    else:
        text = args.prompt if args.prompt is not None else f"<|user|>{args.chat}<|assistant|>"
        ids = tok.encode(text, add_special_tokens=False).ids
    eos = set(store.json("config.json")["eos_token_id"])

    model = Flint(store)
    t0 = time.time()
    trace: dict = {}
    logits = model.forward(ids, trace if args.dump_hidden else None)
    steps = []
    generated: list[int] = []
    for step in range(args.max_new):
        top = torch.topk(logits, args.top)
        nxt = int(top.indices[0])
        steps.append({"top_ids": top.indices.tolist(), "top_logits": [round(v, 4) for v in top.values.tolist()]})
        generated.append(nxt)
        print(f"step {step}: {nxt!r:>7} {tok.decode([nxt])!r}  top={list(zip(top.indices.tolist(), [round(v,2) for v in top.values.tolist()]))}", file=sys.stderr, flush=True)
        if nxt in eos or step == args.max_new - 1:
            break
        logits = model.forward([nxt])
    dt = time.time() - t0
    result = {
        "prompt": text,
        "prompt_ids": ids,
        "generated_ids": generated,
        "generated_text": tok.decode(generated),
        "steps": steps,
        "seconds": round(dt, 1),
        "expert_loads": model.stats["expert_loads"],
    }
    if args.dump_hidden:
        result["hidden_last_prompt_row"] = [[round(v, 5) for v in h.tolist()] for h in trace["hidden_last_row"]]
        result["routing_last_prompt_row"] = trace.get("routing", [])
    print(json.dumps({k: v for k, v in result.items() if not k.startswith(("hidden", "routing", "steps"))}, ensure_ascii=False))
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
