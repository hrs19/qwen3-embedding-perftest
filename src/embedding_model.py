"""Qwen3-Embedding model wrapper: loading (with device/quantization handling), last-token
pooling, and batched frozen-mode encode() with throughput logging.

Qwen3-Embedding is a causal-LM-backbone embedding model: the embedding for a sequence is the
last non-padded token's hidden state (hence left-padding), not a separate pooler head.
"""
import logging
import time

import numpy as np
import psutil
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("qwen_embed_bench.embedding_model")

MODEL_IDS = {
    "0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "4b": "Qwen/Qwen3-Embedding-4B",
    "8b": "Qwen/Qwen3-Embedding-8B",
}
MODEL_PARAMS = {"0.6b": 0.6e9, "4b": 4e9, "8b": 8e9}
QUANTIZED_SIZES = {"4b", "8b"}  # loaded 4-bit on GPU / bf16 on CPU due to memory constraints


class UnsupportedConfigError(RuntimeError):
    """Raised when a requested (model_size, device, mode) combination is not supported."""


class InsufficientMemoryError(RuntimeError):
    """Raised when a preflight memory check determines the run would likely OOM."""


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Standard last-token pooling for left-padded causal-LM embedding models."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def _preflight_check(model_size: str, device: str, mode: str) -> None:
    """Rough required-memory estimate vs available memory; raises with a clear message
    instead of letting the run OOM partway through."""
    params = MODEL_PARAMS[model_size]

    if device == "cuda":
        if model_size in QUANTIZED_SIZES:
            bytes_per_param = 0.6  # 4-bit NF4 + overhead
        else:
            bytes_per_param = 2  # fp16
        required_gb = (params * bytes_per_param) / 1e9 * 1.5  # safety margin for activations
        free_gb, total_gb = torch.cuda.mem_get_info()
        free_gb, total_gb = free_gb / 1e9, total_gb / 1e9
        logger.info("GPU memory: %.2fGB free / %.2fGB total; estimated requirement: %.2fGB", free_gb, total_gb, required_gb)
        if free_gb < required_gb:
            raise InsufficientMemoryError(
                f"GPU has {free_gb:.2f}GB free but this run needs an estimated {required_gb:.2f}GB "
                f"(model_size={model_size}, mode={mode}). Close other GPU processes and retry."
            )
    else:  # cpu
        if model_size in QUANTIZED_SIZES and mode == "lora":
            raise UnsupportedConfigError(
                f"{model_size} + lora + cpu is not supported: no CPU 4-bit quantization path is available, and "
                f"full-precision backprop through a {model_size}-param model on CPU is impractically slow. "
                f"Use --device cuda for {model_size} lora, or --mode frozen for a CPU {model_size} run."
            )
        bytes_per_param = 4 if model_size == "0.6b" else 2  # 0.6b:fp32, 4b/8b:bf16 (memory-constrained)
        required_gb = (params * bytes_per_param) / 1e9 * 1.3
        available_gb = psutil.virtual_memory().available / 1e9
        logger.info("System RAM available: %.2fGB; estimated requirement: %.2fGB", available_gb, required_gb)
        if available_gb < required_gb:
            raise InsufficientMemoryError(
                f"Only {available_gb:.2f}GB RAM free but this run needs an estimated {required_gb:.2f}GB "
                f"(model_size={model_size}, mode={mode}). Close other applications and retry."
            )


class QwenEmbeddingModel:
    def __init__(self, model_size: str, device: str, mode: str):
        if model_size not in MODEL_IDS:
            raise ValueError(f"Unknown model_size {model_size!r}, expected one of {list(MODEL_IDS)}")
        if device not in ("cpu", "cuda"):
            raise ValueError(f"Unknown device {device!r}, expected 'cpu' or 'cuda'")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available on this machine")

        self.model_size = model_size
        self.device = device
        self.mode = mode
        self.model_id = MODEL_IDS[model_size]

        _preflight_check(model_size, device, mode)

        logger.info("Loading tokenizer for %s (padding_side=left)...", self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")

        load_kwargs = {}
        if device == "cuda" and model_size in QUANTIZED_SIZES:
            logger.info("Loading %s on GPU with 4-bit (NF4) quantization", self.model_id)
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = {"": 0}
        elif device == "cuda":
            logger.info("Loading %s on GPU in fp16", self.model_id)
            load_kwargs["dtype"] = torch.float16
        elif model_size in QUANTIZED_SIZES:
            logger.info("Loading %s on CPU in bf16 (memory-constrained)", self.model_id)
            load_kwargs["dtype"] = torch.bfloat16
        else:
            logger.info("Loading %s on CPU in fp32", self.model_id)
            load_kwargs["dtype"] = torch.float32

        t0 = time.time()
        self.model = AutoModel.from_pretrained(self.model_id, **load_kwargs)
        if device == "cuda" and model_size not in QUANTIZED_SIZES:
            self.model = self.model.to(device)
        elif device == "cpu":
            self.model = self.model.to(device)
        logger.info("Model loaded in %.1fs", time.time() - t0)

        if mode == "frozen":
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

        self.hidden_size = self.model.config.hidden_size

    def pooled_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Runs the backbone and returns pooled (last-token) embeddings. Gradient-tracked if
        called outside torch.no_grad() -- used directly by lora_finetune.py."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return last_token_pool(outputs.last_hidden_state, attention_mask)

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        batch_size: int = 16,
        max_length: int = 256,
        dimension: int | None = None,
        normalize: bool = False,
    ) -> np.ndarray:
        """Frozen-mode batched embedding extraction with throughput logging.

        dimension: if set, truncates to the first `dimension` dims (Matryoshka/MRL truncation --
        Qwen3-Embedding is MRL-trained so leading dimensions carry the most signal). Truncation
        happens before normalization, matching Qwen's documented usage.
        normalize: L2-normalize each embedding (standard for cosine-similarity use).
        """
        if dimension is not None and dimension > self.hidden_size:
            logger.warning(
                "Requested dimension=%d exceeds model's native hidden_size=%d; using full size instead",
                dimension, self.hidden_size,
            )
            dimension = None

        self.model.eval()
        all_embeddings = []
        n = len(texts)
        t0 = time.time()
        for start in range(0, n, batch_size):
            batch_texts = texts[start:start + batch_size]
            enc = self.tokenizer(
                batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
            ).to(self.device)
            pooled = self.pooled_forward(enc["input_ids"], enc["attention_mask"]).to(torch.float32)
            if dimension is not None:
                pooled = pooled[:, :dimension]
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.cpu().numpy())

            done = start + len(batch_texts)
            if (start // batch_size) % 10 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                logger.debug("Encoded %d/%d (%.1f samples/sec)", done, n, rate)

        elapsed = time.time() - t0
        logger.info(
            "encode() done: %d samples in %.1fs (%.1f samples/sec) on %s",
            n, elapsed, n / elapsed if elapsed > 0 else 0.0, self.device,
        )
        return np.concatenate(all_embeddings, axis=0)
