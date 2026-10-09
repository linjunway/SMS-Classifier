"""Streamlit demo.   streamlit run app/streamlit_app.py

Two modes:
  * API mode        - if `API_URL` is set (env var or Streamlit secret) every request goes to
                      the FastAPI service (this is what docker-compose does).
  * Standalone mode - otherwise the model is loaded in-process, so the demo also works on
                      Streamlit Community Cloud where no separate API is running.
"""
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


def _secret_api_url():
    try:
        return st.secrets.get("API_URL")
    except FileNotFoundError:  # no secrets file configured
        return None


API_URL = (os.getenv("API_URL") or _secret_api_url() or "").rstrip("/")
STANDALONE = not API_URL
COLORS = {"Normal": "green", "Spam": "orange", "Smishing": "red"}
EXAMPLES = {
    "Friendly message": "Dunno da next show aft 6 is 850. Toa payoh got 650.",
    "Promotional spam": "WINNER!! You have been selected to receive a £900 prize reward! "
                        "To claim call 09061701461. Claim code KL341.",
    "Smishing attempt": "Your parcel is on hold due to unpaid customs fee. "
                        "Pay now at http://dhl-track-pay.xyz/a1 to avoid return.",
}

st.set_page_config(page_title="SMS Smishing Classifier", page_icon="📱", layout="wide")
st.title("📱 SMS Smishing Classifier")
st.caption("Normal vs Spam vs Smishing: Hierarchical Scikit-Learn Ensemble Served Via FastAPI")


@st.cache_resource(show_spinner="Loading model (the first start trains it, ~30 s)...")
def local_classifier():
    # Must be set before `smsclf` is imported so data/model paths resolve to the repo root.
    repo_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("SMSCLF_HOME", str(repo_root))
    if str(repo_root / "src") not in sys.path:  # works without `pip install .`
        sys.path.insert(0, str(repo_root / "src"))
    from smsclf.classifier import SMSClassifier
    from smsclf.config import MODEL_PATH

    if not MODEL_PATH.exists():  # models/*.joblib is git-ignored, so train on first boot
        from smsclf import train

        train.main()
    return SMSClassifier.load(MODEL_PATH)


@st.cache_data(ttl=60)
def get_json(path: str):
    if STANDALONE:
        local_classifier()  # first: sets up sys.path / paths so `smsclf` can be imported
        from smsclf import __version__
        from smsclf.config import METRICS_PATH

        if path == "/health":
            return {"status": "ok (embedded model)", "version": __version__}
        return json.loads(METRICS_PATH.read_text())
    r = requests.get(f"{API_URL}{path}", timeout=5)
    r.raise_for_status()
    return r.json()


def post(path: str, payload: dict):
    if STANDALONE:
        clf = local_classifier()
        if path == "/predict":
            return asdict(clf.predict([payload["text"]])[0])
        preds = clf.predict(payload["texts"])
        return {"count": len(preds), "predictions": [asdict(p) for p in preds]}
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


try:
    health = get_json("/health")
except (requests.RequestException, FileNotFoundError, ImportError) as exc:
    if STANDALONE:
        st.error(f"Could not load the embedded model: {exc}")
    else:
        st.error(f"Cannot reach the API at {API_URL}. Is the service running?")
    st.stop()

with st.sidebar:
    st.subheader("Model performance")
    try:
        info = get_json("/model-info")
        o = info["overall"]
        st.metric("Accuracy", f"{o['accuracy']:.2%}")
        st.metric("Balanced accuracy", f"{o['balanced_accuracy']:.2%}")
        st.metric("Macro F1", f"{o['macro_f1']:.2%}")
        cm = pd.DataFrame(info["confusion_matrix"]["matrix"],
                          index=[f"true {n}" for n in info["confusion_matrix"]["labels"]],
                          columns=info["confusion_matrix"]["labels"])
        st.caption("Confusion matrix (val + test)")
        st.dataframe(cm)
    except (requests.RequestException, FileNotFoundError):
        st.info("Metrics unavailable.")
    mode = "standalone (model loaded in-process)" if STANDALONE else f"API · {API_URL}"
    st.caption(f"v{health['version']} · {mode} · status: {health['status']}")

single, batch = st.tabs(["Single message", "Batch (CSV)"])

with single:
    choice = st.selectbox("Load an example", ["—"] + list(EXAMPLES))
    text = st.text_area("SMS text", value=EXAMPLES.get(choice, ""), height=120)
    if st.button("Classify", type="primary", disabled=not text.strip()):
        res = post("/predict", {"text": text})
        color = COLORS[res["label_name"]]
        st.markdown(f"### :{color}[{res['label_name']}]  ·  confidence {res['confidence']:.1%}")
        c1, c2 = st.columns(2)
        c1.bar_chart(pd.Series(res["probabilities"], name="probability"))
        c2.write("**Detected signals**")
        c2.json(res["signals"])

with batch:
    st.write("Upload a CSV with a `text` column (max 100 rows per request).")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        if "text" not in df.columns:
            st.error("CSV needs a `text` column.")
        else:
            df = df.dropna(subset=["text"]).head(100)
            res = post("/predict/batch", {"texts": df["text"].astype(str).tolist()})
            out = df.assign(
                prediction=[p["label_name"] for p in res["predictions"]],
                confidence=[p["confidence"] for p in res["predictions"]],
            )
            st.dataframe(out, use_container_width=True)
            st.download_button("Download results", out.to_csv(index=False), "predictions.csv")
