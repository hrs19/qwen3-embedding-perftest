"""Embeds a single text and saves the full raw vector so you can inspect what it actually
looks like (shape, dtype, sample values, norm) -- not a benchmark, just a look-and-see tool.

Example:
    python src/inspect_embedding.py --model-size 0.6b --device cpu --dimension default
    python src/inspect_embedding.py --model-size 0.6b --device cuda --dimension 768 --text "hello world"
"""
import argparse
import json
from pathlib import Path

from embedding_model import QwenEmbeddingModel

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "sample_vectors"

DEFAULT_TEXT = "what expression would i use to say i love you if i were an italian"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-size", choices=["0.6b", "4b", "8b"], default="0.6b")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--dimension", default="default", help="Integer (e.g. 768) or 'default' for native size")
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--preview-n", type=int, default=10, help="How many vector values to print")
    return p.parse_args()


def main():
    args = parse_args()
    dimension = None if args.dimension == "default" else int(args.dimension)

    print(f"Loading Qwen3-Embedding-{args.model_size} on {args.device}...")
    embed_model = QwenEmbeddingModel(args.model_size, args.device, mode="frozen")

    embeddings = embed_model.encode([args.text], batch_size=1, dimension=dimension, normalize=True)
    vector = embeddings[0]

    print()
    print(f"Text:              {args.text!r}")
    print(f"Native hidden_size: {embed_model.hidden_size}")
    print(f"Output vector shape: {vector.shape}")
    print(f"Dtype:              {vector.dtype}")
    print(f"L2 norm:            {float((vector ** 2).sum() ** 0.5):.6f}")
    print(f"Min / Max:           {vector.min():.6f} / {vector.max():.6f}")
    print(f"Mean / Std:          {vector.mean():.6f} / {vector.std():.6f}")
    print()
    print(f"First {args.preview_n} values:")
    print(vector[:args.preview_n].tolist())
    print(f"Last {args.preview_n} values:")
    print(vector[-args.preview_n:].tolist())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.model_size}_dim{vector.shape[0]}_{args.device}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "text": args.text,
            "model_size": args.model_size,
            "device": args.device,
            "native_hidden_size": embed_model.hidden_size,
            "output_dimension": int(vector.shape[0]),
            "vector": vector.tolist(),
        }, f, indent=2)

    print()
    print(f"Full vector saved to {out_path}")


if __name__ == "__main__":
    main()
