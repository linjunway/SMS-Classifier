"""Data loading and training-set curation."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .config import EVAL_LABEL_PATH, EVAL_TEXT_PATH, LABELS, TRAIN_PATH
from .preprocessing import build_features


def read_text_data(path: Path) -> tuple[list[str], list[int]]:
    """Read `text,label` rows (label column is optional)."""
    texts, labels = [], []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, delimiter=",", quotechar='"'):
            texts.append(row[0])
            if len(row) > 1:
                labels.append(int(row[1]))
    return texts, labels


def load_train() -> pd.DataFrame:
    texts, labels = read_text_data(TRAIN_PATH)
    return pd.DataFrame({"text": texts, "label": labels})


def curate_train(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve label conflicts, drop exact duplicates and rows that are empty once cleaned.

    * Identical messages carrying different labels are resolved to the most severe label
      (in the provided data this affects one 'click the WAP link' message that impersonates
      a service -> Smishing).
    * Exact duplicates are dropped on the raw text.
    * Messages that are empty after normalisation are removed.
    """
    df = df.copy()
    df["label"] = df.groupby("text")["label"].transform("max")
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    cleaned_empty = build_features(df["text"])["text"].eq("").to_numpy()
    return df.loc[~cleaned_empty].reset_index(drop=True)


def load_eval() -> tuple[list[str], pd.DataFrame]:
    """Held-out messages plus their ground truth (`Id`, `Prediction`, `Usage`)."""
    texts, _ = read_text_data(EVAL_TEXT_PATH)
    truth = pd.read_csv(EVAL_LABEL_PATH)
    return texts, truth


def label_name(label: int) -> str:
    return LABELS[int(label)]
