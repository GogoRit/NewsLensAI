# summarizer/gemini_agent.py

import os
import logging
from typing import List

from bs4 import BeautifulSoup
import google.generativeai as genai
from google.generativeai import GenerationConfig
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# Configure Gemini API
# -----------------------------------------------------------------------------
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise EnvironmentError("Set GOOGLE_API_KEY in .env")
genai.configure(api_key=API_KEY)

# -----------------------------------------------------------------------------
# Logger setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Filter out ad lines
# -----------------------------------------------------------------------------
_AD_KEYWORDS = {"advertisement", "sponsored", "click here", "subscribe"}

def _clean_text(raw: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return " ".join(ln for ln in lines if not any(k in ln.lower() for k in _AD_KEYWORDS))

def summarize_with_gemini(
    raw_text: str,
    entities: List[str],
    max_words: int = 50,
    max_input_chars: int = 4000
) -> str:
    # Clean HTML/ads
    text = _clean_text(raw_text)
    # Truncate if too long
    if len(text) > max_input_chars:
        text = text[-max_input_chars:]

    # Build self‑verifying prompt
    prompt = (
        f"Please summarize the following article in exactly {max_words} words—no more, no less. "
        f"Use these key entities: {', '.join(entities)}. "
        "Count your words and include the count at the end. "
        "Do not add any commentary.\n\n"
        f"Article:\n{text}\n\nSummary:"
    )

    try:
        model  = genai.GenerativeModel(model_name="gemini-1.5-flash")
        cfg    = GenerationConfig(
            max_output_tokens=max_words * 3,
            temperature=0.2,
            top_p=0.9,
            top_k=40
        )
        resp   = model.generate_content(prompt, generation_config=cfg)
        summary = resp.text.strip()
        logger.info("Raw summary: %s", summary)

        # Enforce exact word count
        words = summary.split()
        if len(words) > max_words:
            summary = " ".join(words[:max_words])
            logger.info("Truncated to %d words", max_words)

        return summary

    except Exception as e:
        logger.error("Gemini error: %s", e)
        raise