import numpy as np


def test_predictions_valid(classifier):
    preds = classifier.predict(["see you at 6", "WIN £1000 now call 09061701461 !!!"])
    assert len(preds) == 2
    for p in preds:
        assert p.label in (0, 1, 2)
        assert abs(sum(p.probabilities.values()) - 1) < 1e-2  # chain rule sums to 1 (rounded)


def test_probabilities_sum_to_one(classifier):
    from smsclf.preprocessing import build_features

    proba = classifier.model.predict_proba(build_features(["hello", "FREE prize http://x.co"]))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_empty_after_cleaning_defaults_to_normal(classifier):
    p = classifier.predict(["...!!!"])[0]
    assert p.label == 0 and p.confidence == 1.0


def test_obvious_normal_message(classifier):
    assert classifier.predict(["ok see you at the canteen later"])[0].label == 0
