"""Inference wrapper used by the API: raw strings in, labelled predictions out."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .config import LABELS, MODEL_PATH, NORMAL
from .preprocessing import build_features


@dataclass
class Prediction:
    label: int
    label_name: str
    confidence: float
    probabilities: dict[str, float]
    signals: dict[str, float]


class SMSClassifier:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> SMSClassifier:
        return cls(joblib.load(path))

    def predict(self, texts: list[str]) -> list[Prediction]:
        if not texts:
            return []
        frame = build_features(texts)
        labels = self.model.predict(frame)
        proba = self.model.predict_proba(frame)

        # Messages with nothing left after cleaning (e.g. "...") default to Normal.
        empty = frame["text"].str.strip().eq("").to_numpy()
        labels = np.where(empty, NORMAL, labels)
        proba[empty] = [1.0, 0.0, 0.0]

        out = []
        for i, lab in enumerate(labels):
            row = frame.iloc[i]
            out.append(Prediction(
                label=int(lab),
                label_name=LABELS[int(lab)],
                confidence=round(float(proba[i, lab]), 4),
                probabilities={LABELS[k]: round(float(proba[i, k]), 4) for k in LABELS},
                signals={
                    "urls": int(row["num_urls"]),
                    "phone_numbers": int(row["num_phones"]),
                    "currency_symbols": int(row["num_currency"]),
                    "all_caps_words": int(row["all_caps_words"]),
                    "exclamation_marks": int(row["exclamation_count"]),
                },
            ))
        return out
