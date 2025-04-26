# app.py

import os
import streamlit as st
import pandas as pd
import altair as alt
from dotenv import load_dotenv

from scrapers.guardian_api import fetch_guardian_articles
from summarizers.gemini_agent import summarize_with_gemini

from ner.ner_module import extract_entities
from evaluation.hallucination_module import compute_similarity
from evaluation.bias_module import detect_bias
from evaluation.sentiment_module import analyze_sentiment
from evaluation.toxicity_module import detect_toxicity

# -----------------------------------------------------------------------------
# Page configuration and CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NewsLens AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Base colors and fonts */
    :root {
        --primary-color: #0369A1;
        --background-light: #F3F4F6;
        --text-dark: #1F2937;
        --card-bg: #FFFFFF;
        --muted-text: #4B5563;
    }
    body {
        color: var(--text-dark);
        background-color: var(--background-light);
        font-family: "Helvetica Neue", Arial, sans-serif;
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--primary-color);
        margin-bottom: 0.2rem;
    }
    .header-sub {
        font-size: 1rem;
        color: var(--muted-text);
        margin-bottom: 1.5rem;
    }
    .sidebar .stTextInput > div > input {
        border-radius: 4px;
    }
    .metric-card {
        background-color: var(--card-bg);
        padding: 1rem;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .summary-card {
        background-color: var(--card-bg);
        padding: 1.5rem;
        border-left: 4px solid var(--primary-color);
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: #FFF;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .tab-header {
        font-size: 1.25rem;
        font-weight: 500;
        color: var(--text-dark);
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Load API keys
# -----------------------------------------------------------------------------
load_dotenv()
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY")
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY")
if not GUARDIAN_API_KEY or not GOOGLE_API_KEY:
    st.sidebar.error("Please set GUARDIAN_API_KEY and GOOGLE_API_KEY in .env")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar: search and select
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("NewsLens AI")
    query = st.text_input("Search news", value="abortion rights")
    count = st.slider("Number of articles", 5, 20, 10)
    st.markdown("---")
    try:
        articles = fetch_guardian_articles(query=query, page_size=count)
    except Exception as e:
        st.error(f"Error fetching articles: {e}")
        st.stop()
    if not articles:
        st.warning("No results. Try another query.")
        st.stop()
    titles = [a["title"] for a in articles]
    selected_title = st.selectbox("Select article", titles)
    selected = next(a for a in articles if a["title"] == selected_title)
    st.markdown("---")
    st.caption("Capstone Project • NewsLens AI")

# -----------------------------------------------------------------------------
# Main header
# -----------------------------------------------------------------------------
st.markdown(f"<div class='header-title'>{selected['title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='header-sub'><a href='{selected['link']}' target='_blank'>Read on The Guardian</a></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_preview, tab_analysis = st.tabs(["Preview & Summary", "Analysis"])

with tab_preview:
    st.markdown("<div class='tab-header'>Article Preview</div>", unsafe_allow_html=True)
    full_text = selected["full_text"]
    preview = full_text if len(full_text) <= 500 else full_text[:500].rsplit(" ",1)[0] + "..."
    st.write(preview)
    if len(full_text) > 500:
        st.markdown(f"[Read full article →]({selected['link']})")

    st.markdown("<div class='tab-header'>Generate Summary</div>", unsafe_allow_html=True)
    with st.form("summarize_form"):
        generate = st.form_submit_button("Summarize")
        if generate:
            summary = summarize_with_gemini(raw_text=full_text, entities=[], max_words=50)
            st.markdown(f"<div class='summary-card'>{summary}</div>", unsafe_allow_html=True)

with tab_analysis:
    st.markdown("<div class='tab-header'>Metrics & Insights</div>", unsafe_allow_html=True)
    if 'summary' not in locals():
        summary = summarize_with_gemini(raw_text=full_text, entities=[], max_words=50)

    # run evaluations
    ents       = extract_entities(full_text, summary)
    sim_score  = compute_similarity(full_text, summary)
    bias       = detect_bias(full_text, summary)
    sentiment  = analyze_sentiment(full_text, summary)
    toxicity   = detect_toxicity(full_text, summary)

    # metrics cards
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><strong>Similarity</strong><br><span style='font-size:1.5rem'>{sim_score:.2f}</span></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><strong>Article Bias</strong><br><span style='font-size:1.5rem'>{bias['article_bias']['label']}</span></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><strong>Sentiment</strong><br><span style='font-size:1.5rem'>{sentiment['article_sentiment']['label']}</span></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><strong>Toxicity</strong><br><span style='font-size:1.5rem'>{toxicity['article_toxicity']['label']}</span></div>", unsafe_allow_html=True)

    st.markdown("---")
    # bias chart
    st.subheader("Bias Score Comparison")
    df_bias = pd.DataFrame([
        {"Stage": "Article", "Score": bias['article_bias']['score']},
        {"Stage": "Summary", "Score": bias['summary_bias']['score']}
    ])
    chart_bias = alt.Chart(df_bias).mark_bar().encode(
        x="Stage:N", y=alt.Y("Score:Q", title="Bias Score"),
        color=alt.Color("Stage:N", scale=alt.Scale(range=["#0369A1","#10B981"]))
    ).properties(height=200)
    st.altair_chart(chart_bias, use_container_width=True)

    # sentiment chart
    st.subheader("Sentiment Distribution")
    df_sent = pd.DataFrame([
        {"Stage": "Article", "Sentiment": sentiment['article_sentiment']['label']},
        {"Stage": "Summary", "Sentiment": sentiment['summary_sentiment']['label']}
    ])
    chart_sent = alt.Chart(df_sent).mark_arc(innerRadius=50).encode(
        theta=alt.Theta("count():Q", title=None),
        color=alt.Color("Sentiment:N", legend=alt.Legend(title="Sentiment"))
    ).properties(height=200)
    st.altair_chart(chart_sent, use_container_width=True)

    st.markdown("---")
    st.subheader("Entities Extracted")
    e1, e2 = st.columns(2)
    e1.markdown("**Article Entities**")
    e1.table(ents["article_entities"])
    e2.markdown("**Summary Entities**")
    e2.table(ents["summary_entities"])