"""Text cleaning and meta-feature extraction.

The same functions are used at training and inference time so the two can never drift.

Implemented with Python's `re` module (not `Series.str.*`) on purpose: pandas >= 3 may route
string regexes through Arrow's RE2 engine, whose `\\w` / `\\b` semantics are ASCII-only and
would silently change the features. Plain `re` keeps behaviour identical across versions.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

from .config import META_COLS

_URL = re.compile(r"https?://\S+|www\.\S+")
_PHONE = re.compile(r"\+?\d[\d\-.\s]{6,}\d")
_MONEY = re.compile(r"[\$£€]\d+(?:\.\d+)?")
_NUMBER = re.compile(r"\b\d+\b")
_EMOJI = re.compile("[\U0001F300-\U0001F9FF\U00002600-\U000027BF]")
_NON_WORD = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")

_UPPER = re.compile(r"[A-Z]")
_ALL_CAPS_WORD = re.compile(r"\b[A-Z]{2,}\b")
_DIGIT = re.compile(r"\d")
_URL_MARK = re.compile(r"https?://|www\.")
_CURRENCY = re.compile(r"[\$£€]")


def _clean_one(text: str) -> str:
    text = _URL.sub(" urltoken ", text)
    text = _PHONE.sub(" phonetoken ", text)
    text = _MONEY.sub(" moneytoken ", text)
    text = _NUMBER.sub(" numbertoken ", text)
    text = _EMOJI.sub(" emojitoken ", text)
    text = _NON_WORD.sub(" ", text.lower())
    return _SPACES.sub(" ", text).strip()


def _meta_one(text: str) -> list[float]:
    n_chars = len(text)
    return [
        n_chars,
        len(text.split()),
        len(_UPPER.findall(text)) / (n_chars or 1),
        len(_ALL_CAPS_WORD.findall(text)),
        text.count("!"),
        text.count("?"),
        len(_DIGIT.findall(text)),
        len(_URL_MARK.findall(text)),
        len(_PHONE.findall(text)),
        len(_CURRENCY.findall(text)),
    ]


def clean_text(texts: Iterable[str]) -> pd.Series:
    """Replace URLs / phones / money / numbers / emoji with tokens, lowercase, strip punctuation.

    Order of the substitutions matters and matches the original research notebook.
    """
    return pd.Series([_clean_one(str(t)) for t in texts], dtype="object")


def extract_meta_features(texts: Iterable[str]) -> pd.DataFrame:
    """Urgency / pressure signals computed from the raw (un-normalised) message."""
    return pd.DataFrame([_meta_one(str(t)) for t in texts], columns=META_COLS)


def build_features(texts: Iterable[str]) -> pd.DataFrame:
    """Raw messages -> DataFrame[`text` (cleaned) + META_COLS] expected by the model."""
    texts = [str(t) for t in texts]
    frame = extract_meta_features(texts)
    frame.insert(0, "text", clean_text(texts))
    return frame
