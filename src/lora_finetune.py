"""LoRA fine-tuning mode: adapts the Qwen3-Embedding backbone (attention projections) plus a
classification head, trained jointly end-to-end on the requested device."""
import logging
import time

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from classifier_head import ClassifierHead
from embedding_model import QwenEmbeddingModel
from metrics import compute_classification_metrics

logger = logging.getLogger("qwen_embed_bench.lora_finetune")

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def attach_lora(embed_model: QwenEmbeddingModel) -> None:
    """Wraps embed_model.model in-place with a LoRA adapter over attention projections."""
    model = embed_model.model
    if embed_model.model_size == "4b" and embed_model.device == "cuda":
        logger.info("Preparing 4-bit backbone for k-bit LoRA training")
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    trainable, total = peft_model.get_nb_trainable_parameters()
    logger.info("LoRA attached: %d/%d trainable params (%.3f%%)", trainable, total, 100 * trainable / total)
    embed_model.model = peft_model


def train_lora(
    embed_model: QwenEmbeddingModel,
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    num_labels: int,
    epochs: int = 3,
    batch_size: int = 8,
    max_length: int = 256,
    lr: float = 2e-4,
) -> tuple[dict, float]:
    """Fine-tunes backbone (via LoRA) + a classification head jointly. Returns (metrics, train_time_seconds)."""
    device = embed_model.device
    attach_lora(embed_model)

    head = ClassifierHead(embed_model.hidden_size, num_labels).to(device)
    trainable_params = [p for p in embed_model.model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    logger.info("LoRA fine-tuning on %s for %d epochs (%d train examples, batch_size=%d)", device, epochs, len(train_texts), batch_size)
    t0 = time.time()
    embed_model.model.train()
    head.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        indices = torch.randperm(len(train_texts)).tolist()
        for start in range(0, len(train_texts), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_texts = [train_texts[i] for i in batch_idx]
            batch_labels = torch.tensor([train_labels[i] for i in batch_idx], dtype=torch.long, device=device)

            enc = embed_model.tokenizer(
                batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
            ).to(device)

            optimizer.zero_grad()
            pooled = embed_model.pooled_forward(enc["input_ids"], enc["attention_mask"]).float()
            logits = head(pooled)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        logger.debug("Epoch %d/%d: avg_loss=%.4f", epoch + 1, epochs, epoch_loss / max(n_batches, 1))
    train_time = time.time() - t0
    logger.info("LoRA fine-tuning done in %.2fs", train_time)

    embed_model.model.eval()
    head.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[start:start + batch_size]
            enc = embed_model.tokenizer(
                batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
            ).to(device)
            pooled = embed_model.pooled_forward(enc["input_ids"], enc["attention_mask"]).float()
            preds = head(pooled).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())

    metrics = compute_classification_metrics(test_labels, all_preds)
    logger.info("LoRA eval: accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f", metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"])
    return metrics, train_time
