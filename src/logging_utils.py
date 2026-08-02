"""Shared logging setup: console (INFO) + timestamped file handler (DEBUG) under logs/."""
import logging
import sys
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def setup_logging(run_id: str) -> tuple[logging.Logger, Path]:
    """Configure root logger with console + file handlers. Returns (logger, log_file_path)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"{run_id}.log"

    logger = logging.getLogger("qwen_embed_bench")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger, log_file


def make_run_id(model_size: str, mode: str, device: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{model_size}_{mode}_{device}"
