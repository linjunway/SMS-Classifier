import pandas as pd

from smsclf.config import META_COLS
from smsclf.data import curate_train
from smsclf.preprocessing import build_features, clean_text


def test_entities_are_tokenised():
    out = clean_text(["Call +65 9123 4567 now! Win £500 at http://bit.ly/abc 🎉"])[0]
    assert "phonetoken" in out and "moneytoken" in out
    assert "urltoken" in out and "emojitoken" in out
    assert "http" not in out and "£" not in out


def test_cleaning_is_lowercase_and_punctuation_free():
    assert clean_text(["HELLO,   World!!"])[0] == "hello world"


def test_meta_features_use_raw_text():
    f = build_features(["FREE entry!!! Visit www.win.com"])
    assert f.loc[0, "exclamation_count"] == 3
    assert f.loc[0, "all_caps_words"] == 1
    assert f.loc[0, "num_urls"] == 1
    assert list(f.columns) == ["text", *META_COLS]


def test_empty_string_does_not_crash():
    f = build_features([""])
    assert f.loc[0, "text"] == "" and f.loc[0, "caps_ratio"] == 0


def test_curate_resolves_conflicts_and_duplicates():
    df = pd.DataFrame({"text": ["a b c", "a b c", "hello there", "hello there", "..."],
                       "label": [1, 2, 0, 0, 0]})
    out = curate_train(df)
    assert len(out) == 2  # duplicate removed, empty-after-cleaning removed
    assert out.loc[out["text"] == "a b c", "label"].item() == 2  # most severe label wins
