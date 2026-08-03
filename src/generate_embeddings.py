"""CLI entrypoint: pure embedding generation (inference only, no classifier/fine-tuning).
Benchmarks Qwen3-Embedding throughput across model size / device / output dimension.

Example:
    python src/generate_embeddings.py --model-size 0.6b --device cpu --dimension 768
    python src/generate_embeddings.py --model-size 8b --device cuda --dimension default
    python src/generate_embeddings.py --model-size 0.6b --backend mlx --dimension default   # Apple Silicon
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from data import load_sample_texts
from logging_utils import make_run_id, setup_logging
from metrics import MemoryTracker, timer

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "embeddings"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-size", choices=["0.6b", "4b", "8b"], required=True)
    p.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    p.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="Required for --backend torch. Ignored for --backend mlx (MLX uses Apple Silicon unified memory, no separate device to pick).",
    )
    p.add_argument(
        "--dimension", default="default",
        help="Output vector size: an integer (e.g. 768, 1024) for MRL truncation, or 'default' for the model's native size",
    )
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.backend == "torch" and args.device is None:
        p.error("--device is required when --backend torch")
    return args


def main():
    args = parse_args()
    dimension = None if args.dimension == "default" else int(args.dimension)
    device_label = "mlx" if args.backend == "mlx" else args.device

    dim_label = args.dimension if dimension is None else str(dimension)
    run_id_suffix = f"embed_d{dim_label}"
    run_id = make_run_id(args.model_size, run_id_suffix, device_label)
    logger, log_file = setup_logging(run_id)

    logger.info("=== Run %s ===", run_id)
    logger.info(
        "Config: model_size=%s backend=%s device=%s dimension=%s num_samples=%d batch_size=%d max_length=%d",
        args.model_size, args.backend, device_label, args.dimension, args.num_samples, args.batch_size, args.max_length,
    )
    logger.info("Python=%s Platform=%s", sys.version.split()[0], platform.platform())
    if args.backend == "torch":
        import torch
        logger.info("Torch=%s", torch.__version__)
        if args.device == "cuda" and torch.cuda.is_available():
            logger.info("GPU: %s", torch.cuda.get_device_name(0))

    result = {
        "run_id": run_id,
        "model_size": args.model_size,
        "backend": args.backend,
        "device": device_label,
        "requested_dimension": args.dimension,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }

    try:
        with timer("load_sample_texts"):
            texts = load_sample_texts(args.num_samples, seed=args.seed)
        result["actual_num_samples"] = len(texts)

        with timer("load_model"):
            if args.backend == "mlx":
                from embedding_model_mlx import MLXEmbeddingModel
                embed_model = MLXEmbeddingModel(args.model_size, mode="frozen")
            else:
                from embedding_model import QwenEmbeddingModel
                embed_model = QwenEmbeddingModel(args.model_size, args.device, mode="frozen")
        result["native_hidden_size"] = embed_model.hidden_size

        mem = MemoryTracker(device_label)
        with mem:
            t0 = time.time()
            embeddings = embed_model.encode(
                texts, batch_size=args.batch_size, max_length=args.max_length,
                dimension=dimension, normalize=True,
            )
            elapsed = time.time() - t0

        result["output_dimension"] = embeddings.shape[1]
        result["embed_generation_time_sec"] = elapsed
        result["throughput_samples_per_sec"] = len(texts) / elapsed if elapsed > 0 else None
        norms = np.linalg.norm(embeddings, axis=1)
        result["mean_l2_norm"] = float(norms.mean())
        result["std_l2_norm"] = float(norms.std())
        result["peak_mem_gb"] = mem.peak_gb()
        result["status"] = "success"

    except Exception as e:
        logger.exception("Run failed: %s", e)
        result["status"] = "failed"
        result["error"] = str(e)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{run_id}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Results written to %s", result_path)
    logger.info("Log written to %s", log_file)
    logger.info("=== Run %s: %s ===", run_id, result["status"])

    if result["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
