"""Two-stage hierarchical ensemble (ported from the research notebook).

Stage 1  Normal vs Not-Normal : soft-voting of BernoulliNB, MultinomialNB, LogisticRegression
                                over word features + hand-crafted meta features.
Stage 2  Spam vs Smishing     : stacking of char-n-gram NB, two RBF-SVM "specialists" and a
                                word-level logistic regression (Not-Normal messages only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from .config import META_COLS, RANDOM_STATE

MINIMAL_STOPWORDS = sorted({
    "the", "a", "an", "and", "or", "but", "if", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "should", "would", "could",
})


def _text_plus_meta(vectorizer, scaler, model) -> Pipeline:
    prep = ColumnTransformer(
        [("text", vectorizer, "text"), ("meta", scaler, META_COLS)], remainder="drop"
    )
    return Pipeline([("prep", prep), ("clf", model)])


def build_stage1() -> VotingClassifier:
    bnb = _text_plus_meta(
        CountVectorizer(stop_words=MINIMAL_STOPWORDS, max_features=4500, binary=True),
        Binarizer(threshold=0.0),
        BernoulliNB(alpha=0.05),
    )
    mnb = _text_plus_meta(
        TfidfVectorizer(stop_words=MINIMAL_STOPWORDS, max_features=2000, sublinear_tf=True),
        MinMaxScaler(),  # MultinomialNB needs non-negative inputs
        MultinomialNB(alpha=0.1),
    )
    logreg = _text_plus_meta(
        TfidfVectorizer(stop_words=MINIMAL_STOPWORDS, max_features=2000, sublinear_tf=True),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=4000, C=100.0,
                           random_state=RANDOM_STATE),
    )
    return VotingClassifier(
        [("bnb", bnb), ("mnb", mnb), ("logreg", logreg)], voting="soft", n_jobs=-1
    )


def _char_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=5000,
                           sublinear_tf=True)


def build_stage2() -> StackingClassifier:
    mnb_char = Pipeline([
        ("vec", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=4000,
                                sublinear_tf=True)),
        ("clf", MultinomialNB(alpha=0.1)),
    ])
    spam_spec = Pipeline([
        ("vec", _char_tfidf()),
        ("clf", SVC(kernel="rbf", C=0.05, gamma="scale", class_weight="balanced",
                    probability=True, random_state=RANDOM_STATE)),
    ])
    smishing_spec = Pipeline([
        ("vec", _char_tfidf()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                    probability=True, random_state=RANDOM_STATE)),
    ])
    lr_word = Pipeline([
        ("vec", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=3000,
                                sublinear_tf=True)),
        ("clf", LogisticRegression(class_weight="balanced", C=10.0, max_iter=5000,
                                   random_state=RANDOM_STATE)),
    ])
    return StackingClassifier(
        estimators=[("mnb_char", mnb_char), ("spam_spec", spam_spec),
                    ("smishing_spec", smishing_spec), ("lr_word", lr_word)],
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=2000,
                                           random_state=RANDOM_STATE),
        cv=5,
        n_jobs=-1,
    )


class HierarchicalSMS(ClassifierMixin, BaseEstimator):
    """Normal (0) -> [Spam (1) | Smishing (2)].

    Input is the DataFrame produced by `smsclf.preprocessing.build_features`:
    stage 1 uses the cleaned text and meta features, stage 2 uses the cleaned text only.
    """

    def __init__(self, stage1_pipe=None, stage2_pipe=None):
        self.stage1_pipe = stage1_pipe
        self.stage2_pipe = stage2_pipe

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        self.classes_ = np.array([0, 1, 2])

        self.stage1_pipe_ = clone(self.stage1_pipe).fit(X, (y != 0).astype(int))

        not_normal = y != 0
        self.stage2_pipe_ = clone(self.stage2_pipe).fit(
            X.loc[not_normal, "text"].reset_index(drop=True), y[not_normal]
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X).reset_index(drop=True)
        preds = np.zeros(len(X), dtype=int)
        flagged = self.stage1_pipe_.predict(X) == 1
        if flagged.any():
            preds[flagged] = self.stage2_pipe_.predict(X.loc[flagged, "text"])
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Chain rule: P(Normal)=1-p1, P(Spam)=p1*p2[Spam], P(Smishing)=p1*p2[Smishing].

        Note `predict` follows the hierarchical decision rule (stage 1 first), so in rare
        borderline cases its label can differ from the argmax of these probabilities.
        """
        X = pd.DataFrame(X).reset_index(drop=True)
        p_flagged = self.stage1_pipe_.predict_proba(X)[:, 1]
        p2 = self.stage2_pipe_.predict_proba(X["text"])  # columns: [Spam, Smishing]
        return np.column_stack([1 - p_flagged, p_flagged * p2[:, 0], p_flagged * p2[:, 1]])


def build_model() -> HierarchicalSMS:
    return HierarchicalSMS(stage1_pipe=build_stage1(), stage2_pipe=build_stage2())
