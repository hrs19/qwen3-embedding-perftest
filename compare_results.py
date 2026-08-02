"""Aggregates results/*.json from run_experiment.py runs into a single comparison table,
printed to stdout and logged to logs/compare_<timestamp>.log."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from logging_utils import setup_logging  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

COLUMNS = [
    ("model_size", "Model"),
    ("mode", "Mode"),
    ("device", "Device"),
    ("status", "Status"),
    ("embed_extraction_time_sec", "Embed(s)"),
    ("embed_extraction_samples_per_sec", "Embed/s"),
    ("train_time_sec", "Train(s)"),
    ("total_time_sec", "Total(s)"),
    ("train_peak_mem_gb", "PeakMem(GB)"),
    ("accuracy", "Accuracy"),
    ("macro_f1", "MacroF1"),
]


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main():
    logger, log_file = setup_logging("compare")

    result_files = sorted(RESULTS_DIR.glob("*.json"))
    if not result_files:
        logger.warning("No results found in %s. Run src/run_experiment.py first.", RESULTS_DIR)
        return

    rows = []
    for path in result_files:
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

    rows.sort(key=lambda r: (r.get("model_size", ""), r.get("mode", ""), r.get("device", "")))

    header = [label for _, label in COLUMNS]
    widths = [max(len(label), 10) for _, label in COLUMNS]
    for row in rows:
        for i, (key, _) in enumerate(COLUMNS):
            widths[i] = max(widths[i], len(fmt(row.get(key))))

    def format_row(values):
        return " | ".join(v.ljust(w) for v, w in zip(values, widths))

    lines = [format_row(header), "-+-".join("-" * w for w in widths)]
    for row in rows:
        lines.append(format_row([fmt(row.get(key)) for key, _ in COLUMNS]))

    table = "\n".join(lines)
    logger.info("Comparison across %d run(s):\n%s", len(rows), table)
    logger.info("Log written to %s", log_file)


if __name__ == "__main__":
    main()
