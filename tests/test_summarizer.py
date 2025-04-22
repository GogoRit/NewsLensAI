import os
from dotenv import load_dotenv
import pytest
from summarizers.gemini_agent import summarize_with_gemini

@pytest.fixture(autouse=True)
def load_env():
    load_dotenv()

def test_summarizer_basic():
    text = (
        "The Eiffel Tower, built in 1889 for the World’s Fair, "
        "is one of the most iconic landmarks in Paris, France."
    )
    # allow up to 20 words
    summary = summarize_with_gemini(text, entities=["Eiffel Tower", "Paris"], max_words=20)
    words = summary.split()
    # Basic sanity checks
    assert isinstance(summary, str)
    assert 5 <= len(words) <= 20  # got a non‑empty but bounded summary
    assert "Eiffel" in summary or "Paris" in summary