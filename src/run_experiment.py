"""CLI entrypoint: runs one Qwen3-Embedding classification experiment (one model size,
one mode, one device) end-to-end and writes a results JSON + log file.

Example:
    python src/run_experiment.py --model-size 0.6b --mode frozen --device cpu \
        --train-samples 2000 --test-samples 500 --epochs 10
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

from classifier_head import train_head
from data import load_clinc_oos
from embedding_model import QwenEmbeddingModel
from logging_utils import make_run_id, setup_logging
from lora_finetune import train_lora
from metrics import MemoryTracker, timer

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-size", choices=["0.6b", "4b"], required=True)
    p.add_argument("--mode", choices=["frozen", "lora"], required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--train-samples", type=int, default=2000)
    p.add_argument("--test-samples", type=int, default=500)
    p.add_argument("--epochs", type=int, default=None, help="Default: 10 for frozen, 3 for lora")
    p.add_argument("--batch-size", type=int, default=None, help="Default: 32 for frozen, 8 for lora")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--lr", type=float, default=None, help="Default: 1e-3 for frozen, 2e-4 for lora")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    epochs = args.epochs if args.epochs is not None else (10 if args.mode == "frozen" else 3)
    batch_size = args.batch_size if args.batch_size is not None else (32 if args.mode == "frozen" else 8)
    lr = args.lr if args.lr is not None else (1e-3 if args.mode == "frozen" else 2e-4)

    run_id = make_run_id(args.model_size, args.mode, args.device)
    logger, log_file = setup_logging(run_id)
    logger.info("=== Run %s ===", run_id)
    logger.info(
        "Config: model_size=%s mode=%s device=%s train_samples=%d test_samples=%d epochs=%d batch_size=%d max_length=%d lr=%g",
        args.model_size, args.mode, args.device, args.train_samples, args.test_samples, epochs, batch_size, args.max_length, lr,
    )
    logger.info("Python=%s Torch=%s Platform=%s", sys.version.split()[0], torch.__version__, platform.platform())
    if args.device == "cuda" and torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    result = {
        "run_id": run_id,
        "model_size": args.model_size,
        "mode": args.mode,
        "device": args.device,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": args.max_length,
        "lr": lr,
    }

    try:
        with timer("load_dataset"):
            train_texts, train_labels, test_texts, test_labels, label_names = load_clinc_oos(
                args.train_samples, args.test_samples, seed=args.seed,
            )
        result["num_classes"] = len(label_names)
        result["actual_train_samples"] = len(train_texts)
        result["actual_test_samples"] = len(test_texts)

        with timer("load_model"):
            embed_model = QwenEmbeddingModel(args.model_size, args.device, args.mode)

        if args.mode == "frozen":
            mem = MemoryTracker(args.device)
            with mem, timer("embed_extraction"):
                t_start = time.time()
                train_embeddings = embed_model.encode(train_texts, batch_size=batch_size, max_length=args.max_length)
                test_embeddings = embed_model.encode(test_texts, batch_size=batch_size, max_length=args.max_length)
                extraction_time = time.time() - t_start
            result["embed_extraction_time_sec"] = extraction_time
            result["embed_extraction_samples_per_sec"] = (len(train_texts) + len(test_texts)) / extraction_time
            result["embed_extraction_peak_mem_gb"] = mem.peak_gb()

            head_mem = MemoryTracker(args.device)
            with head_mem:
                metrics, train_time = train_head(
                    train_embeddings, train_labels, test_embeddings, test_labels,
                    num_labels=len(label_names), device=args.device,
                    epochs=epochs, batch_size=batch_size, lr=lr,
                )
            result["train_time_sec"] = train_time
            result["train_peak_mem_gb"] = head_mem.peak_gb()
            result["total_time_sec"] = extraction_time + train_time

        else:  # lora
            mem = MemoryTracker(args.device)
            with mem:
                metrics, train_time = train_lora(
                    embed_model, train_texts, train_labels, test_texts, test_labels,
                    num_labels=len(label_names), epochs=epochs, batch_size=batch_size,
                    max_length=args.max_length, lr=lr,
                )
            result["train_time_sec"] = train_time
            result["train_peak_mem_gb"] = mem.peak_gb()
            result["total_time_sec"] = train_time

        result.update(metrics)
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
