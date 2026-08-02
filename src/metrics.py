"""Timing, memory tracking, and classification metric helpers used across run modes."""
import logging
import os
import time
from contextlib import contextmanager

import psutil
import torch
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger("qwen_embed_bench.metrics")


@contextmanager
def timer(label: str):
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    logger.info("[timing] %s: %.2fs", label, elapsed)


class MemoryTracker:
    """Tracks peak memory usage over a block: CUDA max_memory_allocated on GPU, RSS delta on CPU."""

    def __init__(self, device: str):
        self.device = device
        self._start_rss = None

    def __enter__(self):
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        else:
            self._start_rss = psutil.Process(os.getpid()).memory_info().rss
        return self

    def __exit__(self, *exc):
        return False

    def peak_gb(self) -> float:
        if self.device == "cuda":
            return torch.cuda.max_memory_allocated() / 1e9
        current_rss = psutil.Process(os.getpid()).memory_info().rss
        return max(current_rss - self._start_rss, 0) / 1e9


def compute_classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
