"""Train the model, evaluate on the held-out val/test split, and write artefacts.

    python -m smsclf.train
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
from sklearn import metrics

from . import __version__
from .classifier import SMSClassifier
from .config import LABELS, METRICS_PATH, MODEL_PATH, REPORTS_DIR
from .data import curate_train, load_eval, load_train
from .model import build_model
from .preprocessing import build_features


def _scores(y_true, y_pred) -> dict:
    return {
        "n": len(y_true),
        "accuracy": round(metrics.accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(metrics.balanced_accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(metrics.f1_score(y_true, y_pred, average="macro"), 4),
        "f1_per_class": {
            name: round(metrics.f1_score(y_true, y_pred, labels=[k], average="macro"), 4)
            for k, name in LABELS.items()
        },
    }


def _plot_confusion(cm: np.ndarray) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # plotting is optional (not installed in the slim runtime image)
        print("matplotlib not installed - skipping confusion-matrix plot")
        return

    names = list(LABELS.values())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (val + test)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)


def main() -> None:
    raw = load_train()
    train = curate_train(raw)
    print(f"Training rows: {len(raw)} raw -> {len(train)} after curation")
    print("Class counts:", np.bincount(train["label"]).tolist())

    t0 = time.time()
    model = build_model().fit(build_features(train["text"]), train["label"].to_numpy())
    fit_seconds = time.time() - t0
    print(f"Fitted in {fit_seconds:.0f}s")

    # Evaluate through the exact inference path the API uses.
    texts, truth = load_eval()
    preds = np.array([p.label for p in SMSClassifier(model).predict(texts)])
    y_true = truth["Prediction"].to_numpy()

    cm = metrics.confusion_matrix(y_true, preds, labels=[0, 1, 2])
    report = {
        "model_version": __version__,
        "train_rows": len(train),
        "fit_seconds": round(fit_seconds, 1),
        "overall": _scores(y_true, preds),
        "by_split": {
            u: _scores(y_true[(truth["Usage"] == u).to_numpy()],
                       preds[(truth["Usage"] == u).to_numpy()])
            for u in ("val", "test")
        },
        "confusion_matrix": {"labels": list(LABELS.values()), "matrix": cm.tolist()},
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    _plot_confusion(cm)

    print(metrics.classification_report(y_true, preds, target_names=list(LABELS.values()),
                                        digits=4))
    print(json.dumps(report["overall"], indent=2))
    print("Confusion matrix:\n", cm)


if __name__ == "__main__":
    main()
