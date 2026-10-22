# TurboFlint

Swift and Metal inference for Recursive Flint (40B-A2.5B MoE) on Apple Silicon Macs with 16 GB,
forked from drumih/turbo-fieldfare. The resident core (attention, routers, shared experts,
embeddings, head; about 1.3 GB) is memory-mapped; routed experts (21.3 GB) stream from SSD into a
per-layer slot cache.

## Layout

- `Sources/TurboFieldfare/Flint/`: the Flint runtime.
  - `FlintRunner`: decode.
  - `FlintPrefill`: chunked prompt processing.
  - `FlintExpertCache`: slot cache, async `pread`, and next-layer prefetch.
  - `FlintWeightStore`: mmap of the resident blobs.
  - `FlintTokenizer`: tokenizer and chat format.
- `Sources/TurboFieldfare/Metal/Flint/flint.metal`: the Flint kernels. Decode attention reuses
  upstream `attention_decode_partial`.
- `Sources/TurboFieldfareFormat/FlintOllamaCheckpoint.swift`: indexes the Ollama manifest and
  per-tensor safetensors blobs, and decodes `config.json`.
- `Sources/TurboFlintCLI/`: the `turboflint` command.
- `Scripts/reference/flint_reference.py`: the FP32 layer-by-layer oracle that reads the same
  blobs. `Scripts/reference/check_oracle.py` compares the CLI against stored oracle fixtures.
- Upstream Gemma code (TurboFieldfare runtime, repack, app, server) is still in the tree but is
  not used by `turboflint`.

## Model

The model is read in place from Ollama's store. It comes from `ollama pull flint:40b-a2b-qat-int4`,
which is the `RecursiveSI/flint-int4` MLX checkpoint. Never copy, repack, or re-download it: the
disk has little free space. Forward-pass semantics follow recursive-oss/rec
`engine/src/rec_mlx/model.py`.

## Commands

```bash
swift build -c release --product turboflint
.build/release/turboflint --chat "Explain a hash map." --max-new 128
python3 -I Scripts/reference/check_oracle.py      # greedy tokens vs the FP32 oracle
swift test --filter FlintOllamaCheckpointTests
```

## Rules

- Run `check_oracle.py` after any kernel or runtime change. A divergence is only acceptable where
  the script reports a near-tie.
- Activations must stay FP32. Flint's reference runs in BF16, and FP16 intermediates overflow in
  late layers.
- Size `--slots` for the memory actually free (`memory_pressure -Q`). The slot cache is anonymous
  memory: under pressure macOS swaps it, and decode collapses. Each slot is 1.77 MB, times 47 MoE
  layers.
- Run one model process at a time. Report hardware, macOS, commit, exact command, and the full
  timing footer with any measurement.
