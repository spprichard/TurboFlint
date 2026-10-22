"""Dump the oracle's routing (top-6 ids and weights for every token and MoE layer) plus per-layer
hidden states of the last row, for `turboflint --force-routing ... --dump-hidden ...`.

    python3 -I Scripts/reference/dump_routing.py prompt.txt routing.json

Forcing the oracle's routing separates kernel numerics from MoE routing near-ties: when two
experts score within ~1e-4, any change in summation order can flip the choice.
"""
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import torch
from tokenizers import Tokenizer
import flint_reference as fr
torch.set_grad_enabled(False)
store = fr.BlobStore(fr.DEFAULT_MANIFEST)
tok = Tokenizer.from_file(str(store.json_blobs["tokenizer.json"]))
ids = tok.encode(open(sys.argv[1]).read(), add_special_tokens=False).ids
m = fr.Flint(store)
routing = {}
orig = m.moe
def moe(i, x, trace):
    p = f"model.layers.{i}.mlp."
    scores = torch.sigmoid(x @ m.s.f32(p + "gate.weight").T)
    top = torch.topk(scores, 6, dim=-1)
    w = top.values / (top.values.sum(-1, keepdim=True) + 1e-20)
    s = scores.sort(-1, descending=True).values
    routing[i] = {"ids": top.indices.tolist(), "weights": w.tolist(), "gap67": (s[:, 5] - s[:, 6]).tolist()}
    return orig(i, x, trace)
m.moe = moe
trace = {}
logits = m.forward(ids, trace)
gaps = [g for r in routing.values() for g in r["gap67"]]
print("rows*layers", len(gaps), "with 6th/7th gap < 1e-3:", sum(g < 1e-3 for g in gaps), "< 1e-4:", sum(g < 1e-4 for g in gaps))
json.dump({"routing": {str(k): {"ids": v["ids"], "weights": v["weights"]} for k, v in routing.items()},
           "hidden": [[round(x, 6) for x in h.tolist()] for h in trace["hidden_last_row"]]},
          open(sys.argv[2], "w"))
print("saved")
