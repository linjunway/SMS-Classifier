"""FastAPI service.   uvicorn smsclf.api:app --reload"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from . import __version__
from .classifier import SMSClassifier
from .config import METRICS_PATH, MODEL_PATH

logger = logging.getLogger("smsclf.api")

MAX_SMS_CHARS = 1600
MAX_BATCH = 100


class SMSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_SMS_CHARS,
                      examples=["URGENT! Your bank account is locked. Verify at http://bit.ly/x1"])


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)


class Signals(BaseModel):
    urls: int
    phone_numbers: int
    currency_symbols: int
    all_caps_words: int
    exclamation_marks: int


class PredictionResponse(BaseModel):
    label: int = Field(..., description="0=Normal, 1=Spam, 2=Smishing")
    label_name: str
    confidence: float = Field(..., description="Probability of the predicted class")
    probabilities: dict[str, float]
    signals: Signals


class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.clf = SMSClassifier.load(MODEL_PATH)
        logger.info("Loaded model from %s", MODEL_PATH)
    except FileNotFoundError:
        app.state.clf = None
        logger.error("Model file %s not found - run `python -m smsclf.train`", MODEL_PATH)
    yield


app = FastAPI(
    title="SMS Smishing Classifier",
    description="Classifies an SMS as **Normal**, **Spam** or **Smishing** (SMS phishing) "
                "with a two-stage hierarchical scikit-learn ensemble.",
    version=__version__,
    lifespan=lifespan,
)


def _clf(request: Request) -> SMSClassifier:
    clf = request.app.state.clf
    if clf is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return clf


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request):
    loaded = request.app.state.clf is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded,
                          version=__version__)


@app.get("/model-info", tags=["meta"])
def model_info():
    """Evaluation metrics written by the training script."""
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found")
    return json.loads(METRICS_PATH.read_text())


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(body: SMSRequest, request: Request):
    return asdict(_clf(request).predict([body.text])[0])


@app.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
def predict_batch(body: BatchRequest, request: Request):
    if any(not t.strip() or len(t) > MAX_SMS_CHARS for t in body.texts):
        raise HTTPException(status_code=422,
                            detail=f"Each text must be 1-{MAX_SMS_CHARS} non-blank characters")
    preds = _clf(request).predict(body.texts)
    return BatchResponse(count=len(preds), predictions=[asdict(p) for p in preds])
