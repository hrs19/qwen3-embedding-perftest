"""MLX backend for Qwen3-Embedding, for Apple Silicon Macs (M1/M2/M3/M4) -- an alternative to
embedding_model.py's PyTorch/CUDA backend. Mirrors its interface (hidden_size, encode()) so the
CLI scripts can select either backend via --backend, but is EMBEDDING GENERATION ONLY: there is
no LoRA/classifier training path here (mlx_lm.lora training is not implemented), so this backend
only supports generate_embeddings.py / inspect_embedding.py, not run_experiment.py.

CAVEAT: developed and written without access to Apple Silicon hardware -- it has not been run.
The mlx-community repo IDs below were found via web search, not verified to load cleanly; if one
fails, check https://huggingface.co/mlx-community for an available variant of that model size
(e.g. swap "-mxfp8" for "-8bit") and update MODEL_IDS.

Requires (macOS + Apple Silicon only): pip install mlx mlx-embeddings
"""
import logging
import time

import numpy as np

logger = logging.getLogger("qwen_embed_bench.embedding_model_mlx")

# Community MLX conversions of Qwen3-Embedding. Picked the "-mxfp8" variants since their model
# cards note they were converted specifically with mlx-embeddings; unverified beyond that.
MODEL_IDS = {
    "0.6b": "mlx-community/Qwen3-Embedding-0.6B-mxfp8",
    "4b": "mlx-community/Qwen3-Embedding-4B-mxfp8",
    "8b": "mlx-community/Qwen3-Embedding-8B-mxfp8",
}


class UnsupportedConfigError(RuntimeError):
    """Raised when a requested mode isn't supported by this backend."""


class MLXEmbeddingModel:
    def __init__(self, model_size: str, mode: str = "frozen"):
        if model_size not in MODEL_IDS:
            raise ValueError(f"Unknown model_size {model_size!r}, expected one of {list(MODEL_IDS)}")
        if mode != "frozen":
            raise UnsupportedConfigError(
                "The MLX backend only supports embedding generation (mode='frozen'). "
                "LoRA fine-tuning on MLX is not implemented in this project -- use --backend torch for that."
            )

        try:
            from mlx_embeddings.utils import load
        except ImportError as e:
            raise RuntimeError(
                "mlx-embeddings is not installed, or this isn't Apple Silicon macOS (MLX requires both). "
                "Install with: pip install mlx mlx-embeddings"
            ) from e

        self.model_size = model_size
        self.device = "mlx"
        self.mode = mode
        self.model_id = MODEL_IDS[model_size]

        logger.info("Loading %s via mlx-embeddings...", self.model_id)
        t0 = time.time()
        self.model, self.tokenizer = load(self.model_id)
        logger.info("Model loaded in %.1fs", time.time() - t0)

        self.hidden_size = self.model.config.hidden_size

    def encode(
        self,
        texts: list[str],
        batch_size: int = 16,
        max_length: int = 256,
        dimension: int | None = None,
        normalize: bool = False,
    ) -> np.ndarray:
        """Batched embedding extraction. mlx-embeddings' generate() always returns L2-normalized
        embeddings (there's no documented raw/unnormalized output), so `normalize=False` is not
        honored here -- output is always normalized regardless of that flag.

        dimension: truncates to the first `dimension` dims then re-normalizes (Matryoshka/MRL
        truncation). Mathematically equivalent to truncating before normalization -- truncating
        an already-unit-norm vector and renormalizing gives the same direction.
        """
        from mlx_embeddings import generate

        if dimension is not None and dimension > self.hidden_size:
            logger.warning(
                "Requested dimension=%d exceeds model's native hidden_size=%d; using full size instead",
                dimension, self.hidden_size,
            )
            dimension = None

        all_embeddings = []
        n = len(texts)
        t0 = time.time()
        for start in range(0, n, batch_size):
            batch_texts = texts[start:start + batch_size]
            output = generate(self.model, self.tokenizer, texts=batch_texts)
            batch_embeds = np.array(output.text_embeds).astype(np.float32)

            if dimension is not None:
                batch_embeds = batch_embeds[:, :dimension]
                norms = np.linalg.norm(batch_embeds, axis=1, keepdims=True)
                batch_embeds = batch_embeds / norms

            all_embeddings.append(batch_embeds)

            done = start + len(batch_texts)
            if (start // batch_size) % 10 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                logger.debug("Encoded %d/%d (%.1f samples/sec)", done, n, rate)

        elapsed = time.time() - t0
        logger.info(
            "encode() done: %d samples in %.1fs (%.1f samples/sec) on mlx",
            n, elapsed, n / elapsed if elapsed > 0 else 0.0,
        )
        return np.concatenate(all_embeddings, axis=0)
