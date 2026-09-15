"""Streamlit UI for ModelIO Lab. Run with: streamlit run app.py"""

from __future__ import annotations

import streamlit as st
from langchain_core.output_parsers import PydanticOutputParser

from modelio_lab.benchmark import dependency_footprint, load_sample_reviews, loc_comparison, model_swap, parser_stress_test
from modelio_lab.chains import analyze_review
from modelio_lab.config import PARSER_MODES, available_providers
from modelio_lab.native_baseline import NATIVE_PROVIDERS, analyze_native
from modelio_lab.prompts import build_prompt
from modelio_lab.schemas import ReviewAnalysis, RunResult

st.set_page_config(page_title="ModelIO Lab", page_icon="🧪", layout="wide")
st.title("ModelIO Lab")
st.caption("Run one task through LangChain Model I/O and a plain SDK call, then see what breaks.")

providers = available_providers()
samples = load_sample_reviews()

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Provider", providers, help="Add API keys to .env to unlock more providers.")
    parser_mode = st.radio("Parser mode", PARSER_MODES, index=1)
    max_retries = st.slider("Repair attempts after a parse failure", 0, 3, 1, disabled=parser_mode != "robust")
    if providers == ["fake"]:
        st.info("Only the offline fake model is available. It returns deliberately messy output to exercise the parsers.")


def pick_review(key: str) -> str:
    titles = [s["title"] for s in samples]
    choice = st.selectbox("Sample review", titles + ["Write my own"], key=f"{key}_choice")
    default = next((s["text"] for s in samples if s["title"] == choice), "")
    return st.text_area("Review text", value=default, height=130, key=f"{key}_text_{choice}")


def render_result(res: RunResult) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", "Parsed" if res.ok else "Failed")
    c2.metric("Latency", f"{res.latency_ms:.0f} ms")
    c3.metric("Attempts", res.attempts)
    if res.ok:
        st.json(res.result)
    else:
        st.error(res.error)
    if res.raw_output:
        with st.expander("Raw model output"):
            st.code(res.raw_output, language="text")
    if res.trace:
        with st.expander(f"Request/response trace ({len(res.trace)} events)"):
            st.json(res.trace)


tab_analyze, tab_parsers, tab_swap, tab_prompt, tab_footprint = st.tabs(
    ["Analyze", "Parser stress test", "Model swap", "Prompt inspector", "Footprint"]
)

with tab_analyze:
    review = pick_review("analyze")
    if st.button("Run both pipelines", type="primary", disabled=not review.strip()):
        left, right = st.columns(2)
        with left:
            st.subheader("LangChain Model I/O")
            with st.spinner("Running chain"):
                render_result(analyze_review(review, provider, parser_mode, max_retries))
        with right:
            st.subheader("Native SDK")
            if provider in NATIVE_PROVIDERS:
                with st.spinner("Calling SDK"):
                    render_result(analyze_native(review, provider))
            else:
                st.info(f"The native baseline covers {', '.join(NATIVE_PROVIDERS)}. Pick one of those to compare.")

with tab_parsers:
    st.write("Twelve realistic model outputs, from clean JSON to chatty or truncated text, parsed by both parsers.")
    if st.button("Run stress test"):
        rows = parser_stress_test()
        total = len(rows)
        c1, c2 = st.columns(2)
        c1.metric("Stock PydanticOutputParser", f"{sum(r['strict'] for r in rows)}/{total} parsed")
        c2.metric("Robust parser", f"{sum(r['robust'] for r in rows)}/{total} parsed")
        st.dataframe(rows, width="stretch")

with tab_swap:
    st.write("The unified interface promises a one-line provider swap. Run the same code against several providers.")
    chosen = st.multiselect("Providers", providers, default=providers)
    swap_review = pick_review("swap")
    if st.button("Run swap", disabled=not chosen or not swap_review.strip()):
        with st.spinner("Calling each provider"):
            st.dataframe(model_swap(swap_review, chosen, parser_mode, max_retries), width="stretch")

with tab_prompt:
    st.write("Small and local models get a stricter prompt with few-shot examples. Compare what each provider receives.")
    inspect_provider = st.selectbox("Provider to inspect", ["openai", "anthropic", "groq", "ollama", "fake"])
    instructions = PydanticOutputParser(pydantic_object=ReviewAnalysis).get_format_instructions()
    messages = build_prompt(inspect_provider, instructions).format_messages(review="<your review here>")
    st.caption(f"{len(messages)} messages")
    for msg in messages:
        with st.chat_message("assistant" if msg.type == "ai" else "user" if msg.type == "human" else "system"):
            st.markdown(f"**{msg.type}**")
            st.text(msg.content)

with tab_footprint:
    st.write("Installed dependency trees and lines of code for each pipeline.")
    if st.button("Measure"):
        st.subheader("Transitive dependencies")
        st.dataframe(dependency_footprint(), width="stretch")
        st.subheader("Lines of code")
        st.dataframe(loc_comparison(), width="stretch")
