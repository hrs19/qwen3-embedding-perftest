"""Load and subsample the clinc_oos ("plus" config) intent classification dataset."""
import logging

from datasets import load_dataset

logger = logging.getLogger("qwen_embed_bench.data")

DATASET_NAME = "clinc/clinc_oos"
DATASET_CONFIG = "plus"
TEXT_COL = "text"
LABEL_COL = "intent"


def load_clinc_oos(train_samples: int, test_samples: int, seed: int = 42):
    """Returns (train_texts, train_labels, test_texts, test_labels, label_names)."""
    logger.info("Loading dataset %s (config=%s) from HuggingFace...", DATASET_NAME, DATASET_CONFIG)
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)

    label_names = ds["train"].features[LABEL_COL].names
    num_labels = len(label_names)
    logger.info(
        "Dataset loaded: %d classes, train=%d val=%d test=%d (full sizes)",
        num_labels, len(ds["train"]), len(ds["validation"]), len(ds["test"]),
    )

    train_split = ds["train"].shuffle(seed=seed)
    test_split = ds["test"].shuffle(seed=seed)

    n_train = min(train_samples, len(train_split))
    n_test = min(test_samples, len(test_split))
    if n_train < train_samples or n_test < test_samples:
        logger.warning(
            "Requested train=%d/test=%d exceeds available data; using train=%d/test=%d",
            train_samples, test_samples, n_train, n_test,
        )

    train_split = train_split.select(range(n_train))
    test_split = test_split.select(range(n_test))

    train_texts = train_split[TEXT_COL]
    train_labels = train_split[LABEL_COL]
    test_texts = test_split[TEXT_COL]
    test_labels = test_split[LABEL_COL]

    logger.info(
        "Using train=%d test=%d examples across %d classes",
        len(train_texts), len(test_texts), num_labels,
    )

    return train_texts, train_labels, test_texts, test_labels, label_names


def load_sample_texts(num_samples: int, split: str = "test", seed: int = 42) -> list[str]:
    """Loads plain text samples only (no labels) -- for pure embedding-generation runs."""
    logger.info("Loading %d sample texts from %s (config=%s, split=%s)...", num_samples, DATASET_NAME, DATASET_CONFIG, split)
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    ds = ds.shuffle(seed=seed)
    n = min(num_samples, len(ds))
    if n < num_samples:
        logger.warning("Requested %d samples exceeds available %s data; using %d", num_samples, split, n)
    texts = ds.select(range(n))[TEXT_COL]
    logger.info("Loaded %d sample texts", len(texts))
    return texts
