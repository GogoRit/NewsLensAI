# app.py

import os
import streamlit as st
from dotenv import load_dotenv

from scrapers.guardian_api import fetch_guardian_articles
from ner.entity_extractor import extract_entities
from summarizers.gemini_agent import summarize_with_gemini

# CPU‑only evaluation modules
from evaluation.hallucination_eval import evaluate_hallucination_cpu
from evaluation.bias_eval import evaluate_bias_cpu

# -----------------------------------------------------------------------------
# Load environment variables
# -----------------------------------------------------------------------------
load_dotenv()
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY")
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY")

if not GUARDIAN_API_KEY:
    st.error("Missing GUARDIAN_API_KEY in .env")
    st.stop()
if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY in .env")
    st.stop()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
QUERY                = "abortion rights"
PAGE_SIZE            = 10
MAX_PREVIEW_CHARS    = 500   # how much of the article to preview
SUMMARY_WORD_COUNT   = 50    # desired summary length

# -----------------------------------------------------------------------------
# Streamlit page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NewsLens AI (Guardian)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Sidebar: Fetch Guardian articles
# -----------------------------------------------------------------------------
st.sidebar.title("🔍 Guardian News Feed")
try:
    articles = fetch_guardian_articles(query=QUERY, page_size=PAGE_SIZE)
except Exception as e:
    st.sidebar.error(f"Error fetching Guardian articles: {e}")
    st.stop()

if not articles:
    st.sidebar.warning("No articles found for your query.")
    st.stop()

title_to_article = {a["title"]: a for a in articles}
selected_title   = st.sidebar.selectbox("Select an article:", list(title_to_article))
selected         = title_to_article[selected_title]

# -----------------------------------------------------------------------------
# Main: Preview full article text
# -----------------------------------------------------------------------------
st.header(selected["title"])
st.markdown(f"[Read on The Guardian]({selected['link']})")
st.markdown("---")

full_text = selected["full_text"]

if len(full_text) > MAX_PREVIEW_CHARS:
    preview = full_text[:MAX_PREVIEW_CHARS].rsplit(" ", 1)[0] + "..."
    st.subheader("📰 Article Preview")
    st.write(preview)
    st.markdown(f"[Read the full article…]({selected['link']})")
else:
    st.subheader("📰 Article Text")
    st.write(full_text)

# -----------------------------------------------------------------------------
# Summarization Form
# -----------------------------------------------------------------------------
with st.form("summarize_form"):
    st.markdown(f"### ✍️ Generate a {SUMMARY_WORD_COUNT}-word Summary")
    submitted = st.form_submit_button("Summarize Article")
    if submitted:
        with st.spinner("Running NER, summarization & CPU evaluations…"):
            try:
                # 1. Extract named entities from the full text
                entities     = extract_entities(full_text, min_score=0.8)
                key_entities = [e for e in entities if e.score > 0.8]
                key_texts    = [e.text for e in key_entities]

                # 2. Generate the summary, preserving bias & core facts
                summary = summarize_with_gemini(
                    raw_text=full_text,
                    entities=key_texts,
                    max_words=SUMMARY_WORD_COUNT
                )

                # 3. CPU‑only evaluations
                halluc_score   = evaluate_hallucination_cpu(full_text, summary)
                article_bias   = evaluate_bias_cpu(full_text)
                summary_bias   = evaluate_bias_cpu(summary)

                # 4. Display summary and entities
                st.subheader("✏️ Generated Summary")
                st.write(summary)

                st.subheader("📋 Key Entities (≥80% confidence)")
                st.table([{"Entity": t} for t in key_texts])

                # 5. Show BERTScore consistency
                st.subheader("❗ Hallucination Score (BERTScore)")
                st.metric(label="BERTScore F1", value=f"{halluc_score:.2f}")

                # 6. Show BLEURT proxy for bias
                st.subheader("⚖️ Bias Proxy (BLEURT)")
                bias_table = [
                    {"Type": "Article Bias Proxy", "BLEURT Score": f"{article_bias:.2f}"},
                    {"Type": "Summary Bias Proxy", "BLEURT Score": f"{summary_bias:.2f}"}
                ]
                st.table(bias_table)

            except Exception as err:
                st.error(f"Pipeline error: {err}")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("NewsLens AI • Capstone Project")