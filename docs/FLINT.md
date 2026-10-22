# TurboFlint: Flint on a 16 GB Mac

TurboFlint runs Recursive Flint (40B total, 2.5B active, MoE) on a 16 GB M2 MacBook Air. It keeps
the 1.3 GB shared core memory-mapped and streams the 21.3 GB of routed experts from SSD.
Recursive's own engine, `rec`, needs 48 GB and an M4 Pro or newer. mlx-lm needs the whole 22.6 GB
checkpoint resident.

## How it works

- **Weights in place.** Weights are read straight from Ollama's store: the `flint:40b-a2b-qat-int4`
  tag is the `RecursiveSI/flint-int4` MLX checkpoint, kept as one safetensors blob per tensor.
  Resident blobs are memory-mapped into no-copy Metal buffers. Each routed expert is 1.77 MB:
  nine contiguous ranges across the gate, up, and down blobs. Each range is read with `pread`.
- **Decode, per MoE layer.**
  1. One command buffer runs attention, the router, and the next layer's router, then waits.
  2. The CPU queues reads for missing experts at high priority, and for the next layer's predicted
     experts at normal priority.
  3. The shared expert runs on the GPU while the reads finish.
  4. The routed experts run.
  - The next layer's experts are predicted by running layer L+1's router on layer L's
    post-attention input. It guesses 80% of them.
- **Expert cache.** Each layer has a slot cache: least-frequently-used eviction, ties broken by
  least recently used. Slots are page-aligned, no-copy Metal buffers.
- **Prefill.** Prompts run in chunks of 512 tokens. Projections are batched as int4 GEMMs over
  8-token tiles. Routed experts are grouped by expert, so each expert is read once per chunk. The
  next tile loads while the current one computes.
- **Numerics.** Activations are FP32; the attention query and the K/V cache are FP16. FP16
  activations overflow: Flint's reference runs in BF16, and late-layer expert products exceed the
  FP16 range.

## Correctness

`Scripts/reference/flint_reference.py` is an FP32 reference that processes one layer at a time,
reading the same blobs. It follows rec's `model.py`.

| Check | Result |
| --- | --- |
| Greedy tokens on short prompts (`check_oracle.py`) | Match, except one documented near-tie (top-2 gap 0.011) |
| Per-layer hidden state, short prompt | Within 6e-4 relative |
| 814-token prompt with the oracle's routing forced | Every layer within 1e-3, logits within 0.0012 |
| 814-token prompt with free routing | Logits differ by up to 0.8 |

The free-routing difference is caused by MoE routing near-ties, not by a kernel bug. In one case,
layer 9's sixth- and seventh-ranked experts scored 3e-5 apart. Any change in summation order can
flip that choice, and the flip propagates. Expect this against any reference, Recursive's BF16
engine included. `dump_routing.py` together with `--force-routing` separates the two causes.

## Measurements

These were measured on an Apple M2 MacBook Air with 16 GB, macOS 27.0, Xcode 27.0, and Swift 6.4.
Other apps were running and memory was 61–65% free. Each run is a single measurement with a warm,
uncontrolled page cache. The chat prompt was "Explain how a hash map works, with an example in
Python."

| Workload | Setting | Result | Peak footprint |
| --- | --- | ---: | ---: |
| Decode 256 tokens | 32 slots/layer | 7.05 tok/s | 3.09 GB |
| Decode 256 tokens | 48 slots/layer | 7.86 tok/s | 4.42 GB |
| Prefill 814 tokens | chunk 256 | 23.7 tok/s | 3.08 GB |
| Prefill 814 tokens | chunk 512 | 32.6 tok/s | 3.14 GB |
| Prefill 814 tokens | token-by-token | 7.4 tok/s | — |

At 48 slots, 54% of experts were already cached, 33% were prefetched, and 13% were read on demand.
The engine read 287 MB from SSD per token.

## Where the time goes, and next steps

- **SSD-bound.** Decode reads about 2.3–2.5 GB/s against a measured ceiling of about 3.3 GB/s.
  The levers are fewer bytes and fewer syscalls:
  - **More slots, when memory allows.** At 96 slots and 8.4 GB, decode reached 6.3 tok/s before
    prefetch existed, with 74% hits. But it swapped other apps out. The slot cache is anonymous
    memory, so under memory pressure macOS swaps it out instead of dropping it.
  - **A contiguous per-expert repack.** One 1.77 MB read instead of nine. It needs about 21 GB of
    free disk.
  - **Smarter prefetch.** About 35 prefetched experts per token are never used.
- **GPU clock.** Kernels run about 3x slower inside the decode loop than back-to-back. After
  50 ms idle, the same kernel takes 2.3x longer, because the GPU clocks down while decode waits on
  reads. A spin kernel during waits recovered GPU time but cost the same in I/O, so it was dropped.
- **Prefill.** Each chunk streams nearly every expert once. Larger chunks amortize that, and
  tensor-core QMM (upstream `tensorops.metal`) would cut compute.

## Usage

```bash
swift build -c release --product turboflint --product turboflint-server
.build/release/turboflint --chat "Explain a hash map." --max-new 256 --slots 32
.build/release/turboflint-server --port 8080 --slots 32 --max-context 16384
```

`turboflint-server` serves OpenAI Chat Completions on 127.0.0.1 with streaming and tool calls. It
hides the model's `<think>` reasoning and reuses the cached prompt prefix across turns.
