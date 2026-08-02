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

## Output layout

- `logs/` — per-run log files (console + debug detail), gitignored.
- `results/` — per-run JSON results, gitignored (regenerate by re-running the scripts).
