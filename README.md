# Qwen3-Embedding CPU vs GPU Benchmark

Benchmarks [Qwen3-Embedding](https://huggingface.co/collections/Qwen/qwen3-embedding) (0.6B / 4B / 8B) on CPU vs GPU for two use cases:

1. **Classification** — frozen-embeddings + classifier head, or LoRA fine-tuning the backbone itself, on the [`clinc/clinc_oos`](https://huggingface.co/datasets/clinc/clinc_oos) intent dataset (151 classes).
2. **Pure embedding generation** — inference-only throughput/memory across model size, device, and output dimension (supports Matryoshka/MRL truncation to a custom vector size).

Built and tested on a laptop with an RTX 4060 (8GB VRAM) and 16GB system RAM — the code includes preflight memory checks that fail fast with a clear message instead of OOM-crashing when a config doesn't fit.

## Setup

```
conda create -n qwen3-embed python=3.11
conda activate qwen3-embed
pip install -r requirements.txt
```

Uses `transformers>=4.51` (required for Qwen3 architecture support), `peft` + `bitsandbytes` for LoRA/4-bit quantization, `datasets` for the benchmark dataset.

## Model size vs precision

| Model | GPU | CPU |
|---|---|---|
| 0.6B | fp16 | fp32 |
| 4B | 4-bit (NF4, bitsandbytes) | bf16 |
| 8B | 4-bit (NF4, bitsandbytes) | bf16 |

4-bit quantization is GPU-only (no CPU kernel path in bitsandbytes), which is why 4B/8B need much more RAM on CPU than VRAM on GPU — 8B on CPU (~20GB required) will not fit on a 16GB-RAM machine at all.

## Benchmark summary (CPU vs GPU)

Measured on the reference hardware above (RTX 4060 Laptop, 8GB VRAM / 16GB RAM), pure embedding generation, 500 samples from `clinc/clinc_oos`, native output dimension, `torch` backend. See [`src/generate_embeddings.py`](src/generate_embeddings.py) usage above to reproduce; full run matrix and raw numbers in `results/embeddings/comparison.xlsx` after running [`run_embedding_comparison.ps1`](run_embedding_comparison.ps1).

| Model | Params | Device | Samples/sec | Time (500 samples) | Peak memory | GPU speedup |
|---|---|---|---|---|---|---|
| 0.6B | ~0.6B | CPU | 18.9 | 26.5s | 0.21GB | — |
| 0.6B | ~0.6B | GPU | 378.9 | 1.3s | 1.26GB | ~20x |
| 4B | ~4B | CPU | *failed — insufficient RAM* | — | — | — |
| 4B | ~4B | GPU | 73.0 | 6.8s | 2.78GB | n/a (no CPU baseline) |
| 8B | ~8B | CPU | *failed — insufficient RAM* | — | — | — |
| 8B | ~8B | GPU | 38.0 | 13.2s | 4.88GB | n/a (no CPU baseline) |

Takeaways from this hardware:
- GPU throughput drops roughly in line with model size (0.6B → 4B → 8B is ~6.7x → ~2x per step down in speed), consistent with more parameters per forward pass, plus 4B/8B running 4-bit quantized (extra dequant overhead) vs 0.6B's plain fp16.
- CPU only completed the 0.6B model — 4B and 8B need bf16 weights on CPU (no 4-bit CPU kernel), which exceeds available system RAM on a 16GB machine. This is a hardware ceiling, not a code limitation — see the "Model size vs precision" table above.
- These are single-machine, single-run numbers (no repeated trials/averaging) — treat them as directional, not authoritative benchmarks.

## Scripts

### Classification benchmark
```
python src/run_experiment.py --model-size 0.6b --mode frozen --device cpu --train-samples 2000 --test-samples 500
python src/run_experiment.py --model-size 0.6b --mode lora --device cuda --train-samples 2000 --test-samples 500
python compare_results.py   # aggregates results/*.json into a comparison table
```
`--mode frozen` trains a linear head on top of frozen embeddings. `--mode lora` LoRA-finetunes the backbone (attention projections) + head jointly. Not every (model_size, mode, device) combination is supported — e.g. 4B/8B + lora + cpu is blocked outright (no viable path), see `src/embedding_model.py`.

### Pure embedding generation
```
python src/generate_embeddings.py --model-size 0.6b --device cuda --dimension 768
python src/generate_embeddings.py --model-size 8b --device cuda --dimension default
python compare_embedding_results.py   # aggregates results/embeddings/*.json + writes comparison.xlsx
```
`--dimension` truncates the output vector (Matryoshka/MRL — Qwen3-Embedding is trained so leading dimensions carry the most signal); `default` returns the model's native size (1024 / 2560 / 4096 for 0.6B / 4B / 8B respectively).

Run the full model x device x dimension matrix in one go:
```
.\run_embedding_comparison.ps1
```

### Inspect a single vector
```
python src/inspect_embedding.py --model-size 0.6b --device cuda --text "your sentence here"
```
Prints shape/norm/sample values and saves the full vector to `results/sample_vectors/`.

## MLX backend (Apple Silicon)

> ⚠️ **This backend was written entirely by Claude (AI-generated) and has never been run.** The project was developed on a Windows/CUDA machine with no Apple Silicon hardware available to test against. Treat `src/embedding_model_mlx.py` as an unverified first draft — expect to debug it on first use, not just run it.

`generate_embeddings.py` and `inspect_embedding.py` support `--backend mlx` for running on Apple Silicon Macs (M1/M2/M3/M4) via [`mlx-embeddings`](https://github.com/Blaizzy/mlx-embeddings), using community-converted weights from [`mlx-community`](https://huggingface.co/mlx-community). `--device` is ignored in this mode — MLX uses unified memory, there's no separate CPU/GPU split to pick.

```
pip install mlx mlx-embeddings   # macOS + Apple Silicon only
python src/generate_embeddings.py --model-size 0.6b --backend mlx --dimension default
```

**Scope and caveats:**
- **Embedding generation only.** There is no MLX path for `run_experiment.py` (classification / LoRA fine-tuning) — that would need a separate implementation on top of `mlx_lm.lora`, which isn't built here.
- **Unverified model repo IDs.** The `mlx-community` repo IDs in `src/embedding_model_mlx.py` (`*-mxfp8` variants) were found via web search, not confirmed to actually load — if one fails, check [huggingface.co/mlx-community](https://huggingface.co/mlx-community) for an available variant of that model size (e.g. `*-8bit`) and edit `MODEL_IDS`.
- mlx-embeddings' `generate()` always returns L2-normalized output; there's no documented way to get raw unnormalized embeddings through it (unlike the torch backend, which can return either).
- If you get this working (or fix it), consider it a good candidate for a PR back to this repo.

## Output layout

- `logs/` — per-run log files (console + debug detail), gitignored.
- `results/` — per-run JSON results, gitignored (regenerate by re-running the scripts).
