"""Central configuration: labels, feature names and file locations."""
from __future__ import annotations

import os
from pathlib import Path

# Project root. Override with SMSCLF_HOME when the package is installed (e.g. in Docker).
ROOT = Path(os.getenv("SMSCLF_HOME", Path(__file__).resolve().parents[2]))

DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
EVAL_TEXT_PATH = DATA_DIR / "val_test_texts.csv"
EVAL_LABEL_PATH = DATA_DIR / "val_test_labels.csv"

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", ROOT / "reports" / "metrics.json"))
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42

LABELS = {0: "Normal", 1: "Spam", 2: "Smishing"}
NORMAL, SPAM, SMISHING = 0, 1, 2

# Hand-crafted features computed on the RAW text (before normalisation).
META_COLS = [
    "char_count",
    "word_count",
    "caps_ratio",
    "all_caps_words",
    "exclamation_count",
    "question_count",
    "digit_count",
    "num_urls",
    "num_phones",
    "num_currency",
]
