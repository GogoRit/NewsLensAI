import streamlit as st
import pandas as pd
from src.summarization import generate_summary
from src.hallucination import check_hallucination
from src.bias_analysis import analyze_bias

st.title("AI News Summarization Tool")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Process each article
    summaries, hallucination_checks, biases = [], [], []

    for article in df['cleaned_article']:
        summary = generate_summary(article)
        hallucination = check_hallucination(article, summary)
        bias = analyze_bias(summary)

        summaries.append(summary)
        hallucination_checks.append(hallucination)
        biases.append(bias)

    df['summary'] = summaries
    df['hallucination_check'] = hallucination_checks
    df['bias_analysis'] = biases

    st.write("Processed Data:")
    st.dataframe(df[['title', 'summary', 'hallucination_check', 'bias_analysis']])

    # Save to CSV
    df.to_csv("processed_results.csv", index=False)
    st.download_button("Download Processed CSV", "processed_results.csv")