"""Aggregates results/embeddings/*.json from generate_embeddings.py runs into a comparison
table (stdout + log) and an Excel workbook (results/embeddings/comparison.xlsx)."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from logging_utils import setup_logging  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "embeddings"
EXCEL_PATH = RESULTS_DIR / "comparison.xlsx"

COLUMNS = [
    ("model_size", "Model"),
    ("backend", "Backend"),
    ("device", "Device"),
    ("requested_dimension", "ReqDim"),
    ("output_dimension", "OutDim"),
    ("native_hidden_size", "NativeDim"),
    ("status", "Status"),
    ("embed_generation_time_sec", "Time(s)"),
    ("throughput_samples_per_sec", "Samples/s"),
    ("peak_mem_gb", "PeakMem(GB)"),
    ("mean_l2_norm", "MeanNorm"),
]


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main():
    logger, log_file = setup_logging("compare_embed")

    result_files = sorted(RESULTS_DIR.glob("*.json"))
    if not result_files:
        logger.warning("No results found in %s. Run src/generate_embeddings.py first.", RESULTS_DIR)
        return

    rows = []
    for path in result_files:
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
        row.setdefault("backend", "torch")  # older results predate --backend; they were all torch
        rows.append(row)

    rows.sort(key=lambda r: (r.get("model_size", ""), r.get("backend", ""), r.get("device", ""), str(r.get("requested_dimension", ""))))

    header = [label for _, label in COLUMNS]
    widths = [max(len(label), 8) for _, label in COLUMNS]
    for row in rows:
        for i, (key, _) in enumerate(COLUMNS):
            widths[i] = max(widths[i], len(fmt(row.get(key))))

    def format_row(values):
        return " | ".join(v.ljust(w) for v, w in zip(values, widths))

    lines = [format_row(header), "-+-".join("-" * w for w in widths)]
    for row in rows:
        lines.append(format_row([fmt(row.get(key)) for key, _ in COLUMNS]))

    table = "\n".join(lines)
    logger.info("Embedding-generation comparison across %d run(s):\n%s", len(rows), table)

    df = pd.DataFrame(rows)
    dim_order = ["768", "1024", "default"]
    df["requested_dimension"] = pd.Categorical(df["requested_dimension"].astype(str), categories=dim_order, ordered=True)
    df = df.sort_values(["model_size", "backend", "device", "requested_dimension"])

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="raw_results", index=False)

        ok = df[df["status"] == "success"]
        if not ok.empty:
            throughput_pivot = ok.pivot_table(
                index=["model_size", "backend", "device"], columns="requested_dimension",
                values="throughput_samples_per_sec", observed=True,
            )
            throughput_pivot.to_excel(writer, sheet_name="throughput_samples_per_sec", merge_cells=False)

            mem_pivot = ok.pivot_table(
                index=["model_size", "backend", "device"], columns="requested_dimension",
                values="peak_mem_gb", observed=True,
            )
            mem_pivot.to_excel(writer, sheet_name="peak_mem_gb", merge_cells=False)

    logger.info("Excel report written to %s", EXCEL_PATH)
    logger.info("Log written to %s", log_file)


if __name__ == "__main__":
    main()
