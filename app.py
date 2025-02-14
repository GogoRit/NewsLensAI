import streamlit as st
from preprocessing import preprocess_text
from ner import extract_entities
from agents.summarization_agent import SummarizationAgent
from agents.bias_detection_agent import BiasDetectionAgent
from agents.similarity_agent import SimilarityAgent

# Title and description
st.title("Article Summarization Tool with Agentic AI")
st.write("Upload an article, and we'll summarize it using advanced AI agents powered by LLaMA-3 and Phi Data.")

# User input: Upload or paste text
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
article_text = st.text_area("Or paste your article here")

if uploaded_file is not None:
    article_text = uploaded_file.read().decode("utf-8")

if not article_text:
    st.warning("Please upload a file or paste the article text.")
    st.stop()

# Preprocess the text
cleaned_text = preprocess_text(article_text)

# Extract entities
entities = extract_entities(cleaned_text)
st.write("Extracted Entities:", entities)

# Summarize the text
summarization_agent = SummarizationAgent()
summary = summarization_agent.run(cleaned_text)
st.subheader("Summary")
st.write(summary)

# Detect bias
bias_detection_agent = BiasDetectionAgent()
original_bias = bias_detection_agent.run(cleaned_text)
summary_bias = bias_detection_agent.run(summary)
st.subheader("Bias Detection")
st.write(f"Original Article Bias: {original_bias}")
st.write(f"Summary Bias: {summary_bias}")

# Calculate similarity
similarity_agent = SimilarityAgent()
similarity_score = similarity_agent.run(cleaned_text, summary)
st.subheader("Similarity Analysis")
st.write(f"Similarity between Original and Summary: {similarity_score:.2f}")

# Display results
st.subheader("Results")
st.write("### Original Article")
st.write(cleaned_text)

st.write("### Summary")
st.write(summary)

st.write("### Extracted Entities")
st.write(entities)

st.write("### Bias Detection")
st.write(f"Original Article Bias: {original_bias}")
st.write(f"Summary Bias: {summary_bias}")

st.write("### Similarity Analysis")
st.write(f"Similarity Score: {similarity_score:.2f}")