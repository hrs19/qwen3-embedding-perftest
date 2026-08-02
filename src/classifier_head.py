"""Frozen-embeddings mode: a lightweight linear classifier head trained on top of
precomputed Qwen3-Embedding vectors, trained/evaluated on the requested device."""
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from metrics import compute_classification_metrics

logger = logging.getLogger("qwen_embed_bench.classifier_head")


class ClassifierHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        return self.linear(x)


def train_head(
    train_embeddings: np.ndarray,
    train_labels: list[int],
    test_embeddings: np.ndarray,
    test_labels: list[int],
    num_labels: int,
    device: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> tuple[dict, float]:
    """Trains a linear head on the given device. Returns (metrics_dict, train_time_seconds)."""
    hidden_size = train_embeddings.shape[1]
    head = ClassifierHead(hidden_size, num_labels).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.tensor(train_embeddings, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    logger.info("Training classifier head on %s for %d epochs (%d train examples)", device, epochs, len(train_labels))
    t0 = time.time()
    head.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)
        logger.debug("Epoch %d/%d: loss=%.4f", epoch + 1, epochs, epoch_loss)
    train_time = time.time() - t0
    logger.info("Head training done in %.2fs", train_time)

    head.eval()
    with torch.no_grad():
        test_x = torch.tensor(test_embeddings, dtype=torch.float32).to(device)
        preds = head(test_x).argmax(dim=1).cpu().numpy()

    metrics = compute_classification_metrics(test_labels, preds)
    logger.info("Head eval: accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f", metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"])
    return metrics, train_time
