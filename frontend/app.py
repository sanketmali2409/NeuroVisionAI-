"""
NeuroVision AI - Streamlit frontend entrypoint.

This is the landing/dashboard page. Additional pages live in `frontend/pages/`
and are picked up automatically by Streamlit's multipage app mechanism
(files there appear in the sidebar in alphabetical/numeric-prefix order).

Run with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

# Allow `from config.settings import settings` when run as `streamlit run
# frontend/app.py` from the project root, where the project root may not
# already be on sys.path the way it is when running via a package/module.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402

st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal dark-theme polish beyond Streamlit's built-in dark mode toggle.
st.markdown(
    """
    <style>
        .metric-card {
            background-color: #1e1e2e;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            border: 1px solid #313244;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_backend_url() -> str:
    return f"http://localhost:{settings.BACKEND_PORT}{settings.API_V1_PREFIX}"


def check_backend_status() -> dict | None:
    """Ping the backend's health endpoint; returns None if unreachable."""
    try:
        response = requests.get(f"{get_backend_url()}/health", timeout=2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def main() -> None:
    st.title("🧠 NeuroVision AI")
    st.caption("AI-powered brain tumor diagnosis, segmentation, and clinical decision support")

    status = check_backend_status()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if status:
            st.metric("Backend status", "Online", delta="healthy", delta_color="normal")
        else:
            st.metric("Backend status", "Offline")
            st.caption(f"Could not reach the API at {get_backend_url()}. Start it with `uvicorn backend.main:app --reload`.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Environment", settings.APP_ENV)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Classes supported", len(settings.CLASS_NAMES))
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Get started")
    st.markdown(
        """
        Use the sidebar to navigate:

        - **Upload & Predict** — upload an MRI, get classification + confidence
        - **Segmentation** — view the tumor mask overlay
        - **Explainability** — Grad-CAM / SHAP heatmaps
        - **Medical Report** — generate and download a structured report
        - **Chat Assistant** — ask medical questions (RAG-grounded)
        - **Prognosis** — recurrence risk and survival estimates

        *(Pages are added incrementally as each backend module ships — see
        `docs/ROADMAP.md`.)*
        """
    )

    st.warning(
        "Research / portfolio project — not a certified medical device. "
        "Do not use for real clinical decision-making.",
        icon="⚠️",
    )


if __name__ == "__main__":
    main()
