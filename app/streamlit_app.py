"""Streamlit demo that talks to the FastAPI service.   streamlit run app/streamlit_app.py"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
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


@st.cache_data(ttl=60)
def get_json(path: str):
    r = requests.get(f"{API_URL}{path}", timeout=5)
    r.raise_for_status()
    return r.json()


def post(path: str, payload: dict):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


try:
    health = get_json("/health")
except requests.RequestException:
    st.error(f"Cannot reach the API at {API_URL}. Start it with `docker compose up`.")
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
    except requests.RequestException:
        st.info("Metrics unavailable.")
    st.caption(f"API v{health['version']} · status: {health['status']}")

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
