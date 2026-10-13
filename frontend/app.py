"""Streamlit dashboard. Talks to the API over HTTP only.

Run (second terminal):  streamlit run frontend/app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from api_client import ApiClient, ApiError
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def load_feature_meta(api_url: str) -> dict:
    return ApiClient(api_url).model_features()


def history_to_frame(items: list[dict], currency: str) -> pd.DataFrame:
    rows = []
    for it in items:
        rows.append(
            {
                "ID": it["prediction_id"],
                "Created (UTC)": it["created_at"],
                f"Price ({currency})": it["predicted_price"],
                **it["features"],
            }
        )
    return pd.DataFrame(rows)


def validate_inputs(values: dict[str, float], features: list[dict]) -> list[str]:
    errors = []
    for f in features:
        v = values[f["name"]]
        if f.get("minimum") is not None and v < f["minimum"]:
            errors.append(f"{f['title']} must be at least {f['minimum']}.")
        if f.get("exclusive_minimum") is not None and v <= f["exclusive_minimum"]:
            errors.append(f"{f['title']} must be greater than {f['exclusive_minimum']}.")
        if f.get("maximum") is not None and v > f["maximum"]:
            errors.append(f"{f['title']} must be at most {f['maximum']}.")
    return errors


# ------------------------------------------------------------------ sidebar
st.title("🏠 House Price Predictor")
st.caption("California block-group house value estimates, saved to SQL Server through the API.")

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("API base URL", os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
    client = ApiClient(api_url)
    if st.button("Check API status"):
        try:
            client.health()
            st.success("API: online")
        except ApiError as exc:
            st.error(exc.message)
        try:
            client.health_database()
            st.success("Database: reachable")
        except ApiError as exc:
            st.warning(f"Database: {exc.message}")

try:
    meta = load_feature_meta(api_url)
except ApiError as exc:
    st.error(exc.message)
    st.info("Start the API first:  `uvicorn app.main:app --reload`")
    st.stop()

features: list[dict] = meta["features"]
currency: str = meta["currency"]

tab_predict, tab_history, tab_lookup = st.tabs(["Predict", "History", "Look up / delete"])

# ------------------------------------------------------------------ predict
with tab_predict:
    with st.form("predict_form"):
        cols = st.columns(2)
        values: dict[str, float] = {}
        for i, f in enumerate(features):
            lo = f.get("minimum") if f.get("minimum") is not None else f.get("exclusive_minimum")
            with cols[i % 2]:
                values[f["name"]] = st.number_input(
                    f["title"],
                    min_value=float(lo) if lo is not None else None,
                    max_value=float(f["maximum"]) if f.get("maximum") is not None else None,
                    value=float(f["example"]) if f.get("example") is not None else float(lo or 0),
                    format="%.4f",
                    help=f["description"],
                    key=f"in_{f['name']}",
                )
        submitted = st.form_submit_button("Predict House Price", type="primary")

    if submitted:
        problems = validate_inputs(values, features)
        if problems:
            for p in problems:
                st.error(p)
        else:
            try:
                with st.spinner("Predicting and saving..."):
                    result = client.predict(values)
                st.session_state["last_result"] = result
                st.session_state.pop("history", None)  # refresh history next time
            except ApiError as exc:
                st.error(exc.message)

    result = st.session_state.get("last_result")
    if result:
        st.metric(f"Estimated price ({result['currency']})", f"{result['predicted_price']:,.0f}")
        st.caption(f"Saved as prediction #{result['prediction_id']} at {result['created_at']}")

# ------------------------------------------------------------------ history
with tab_history:
    c1, c2, c3 = st.columns([1, 1, 1])
    page_size = c1.selectbox("Rows per page", [10, 20, 50, 100], index=1)
    page = c2.number_input("Page", min_value=1, value=1, step=1)
    c3.write("")
    refresh = c3.button("Refresh history")

    key = (page_size, page)
    if refresh or "history" not in st.session_state or st.session_state.get("history_key") != key:
        try:
            with st.spinner("Loading history..."):
                st.session_state["history"] = client.history(page_size, (page - 1) * page_size)
                st.session_state["history_key"] = key
        except ApiError as exc:
            st.session_state.pop("history", None)
            st.error(exc.message)

    data = st.session_state.get("history")
    if data:
        st.write(f"{data['total']} saved predictions")
        if data["items"]:
            st.dataframe(history_to_frame(data["items"], currency), use_container_width=True, hide_index=True)
        else:
            st.info("No predictions on this page.")

# ------------------------------------------------------------------ lookup
with tab_lookup:
    pid = st.number_input("Prediction ID", min_value=1, value=1, step=1)
    if st.button("View prediction"):
        try:
            st.session_state["viewed"] = client.get_prediction(int(pid))
        except ApiError as exc:
            st.session_state.pop("viewed", None)
            st.error("No prediction with that ID." if exc.status_code == 404 else exc.message)

    viewed = st.session_state.get("viewed")
    if viewed:
        st.metric(f"Estimated price ({viewed['currency']})", f"{viewed['predicted_price']:,.0f}")
        st.json(viewed["features"])
        confirm = st.checkbox("I want to delete this prediction")
        if st.button("Delete", disabled=not confirm):
            try:
                client.delete_prediction(viewed["prediction_id"])
                st.session_state.pop("viewed", None)
                st.session_state.pop("history", None)
                st.success("Deleted.")
            except ApiError as exc:
                st.error(exc.message)
