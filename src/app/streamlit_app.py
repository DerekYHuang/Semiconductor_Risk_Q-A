"""
Local Streamlit demo for the contamination/vapor-intrusion RAG system.

Run:
    streamlit run src/app/streamlit_app.py

Purely local — no cloud deployment needed for a portfolio project. Screen-record or
screenshot this for your writeup.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1] / "retrieval"))
from rag_pipeline import RAGPipeline  # noqa: E402

st.set_page_config(page_title="Silicon Valley Contamination Q&A", layout="centered")

st.title("Silicon Valley Semiconductor Contamination & Vapor-Intrusion Q&A")
st.caption(
    "Locally-run RAG system over EPA Superfund, DTSC EnviroStor, and RWQCB GeoTracker "
    "records for Santa Clara County. Answers are grounded in retrieved source documents — "
    "see the expandable source panel below each answer."
)


@st.cache_resource
def get_pipeline():
    return RAGPipeline()


with st.spinner("Loading pipeline (embedding model + vector store)..."):
    pipeline = get_pipeline()

question = st.text_input(
    "Ask a question about a site, contaminant, or vapor mitigation status:",
    placeholder="e.g. What contaminant was detected at [site] and what year?",
)

if question:
    with st.spinner("Retrieving and generating..."):
        result = pipeline.query(question)

    st.markdown("### Answer")
    st.write(result["answer"])

    with st.expander("Retrieved source chunks"):
        for chunk in result["retrieved_chunks"]:
            st.markdown(f"**{chunk['source_doc']}** (distance: {chunk['distance']:.3f})")
            st.text(chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else ""))
            st.divider()

st.sidebar.markdown("### About this project")
st.sidebar.write(
    "Built to explore semiconductor manufacturing's environmental legacy in Santa Clara "
    "County — companion to the Data Center Energy & Water Footprint Tracker project, "
    "looking at the same industry's footprint from a different angle."
)
