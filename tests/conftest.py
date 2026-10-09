"""Fixtures: a small model is trained once per session on a stratified subset of the real data."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from smsclf.classifier import SMSClassifier
from smsclf.data import curate_train, load_train
from smsclf.model import build_model
from smsclf.preprocessing import build_features


@pytest.fixture(scope="session")
def classifier() -> SMSClassifier:
    df = curate_train(load_train())
    sub = pd.concat(
        [df[df["label"] == k].sample(min((df["label"] == k).sum(), 150), random_state=0)
         for k in (0, 1, 2)]
    ).reset_index(drop=True)
    model = build_model().fit(build_features(sub["text"]), sub["label"].to_numpy())
    return SMSClassifier(model)


@pytest.fixture(scope="session")
def client(classifier):
    from smsclf.api import app

    with TestClient(app) as c:
        app.state.clf = classifier  # override whatever lifespan loaded
        yield c
