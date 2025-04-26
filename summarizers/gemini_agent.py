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
    max_words: int = 100,
    max_input_chars: int = 4000
) -> str:
    # 1. Clean HTML/ads
    text = _clean_text(raw_text)

    # 2. Truncate if too long
    if len(text) > max_input_chars:
        text = text[-max_input_chars:]  # use the last chunk for context

    # 3. Build an editorial-style, approximate-length prompt
    entities_list = ", ".join(entities) if entities else "None"
    prompt = f"""
    You are an experienced news editor. Summarize the following article into one concise, coherent paragraph of around {max_words} words:

    • Use complete sentences—do not cut off mid-thought.  
    • Include all Important Entities (confidence ≥ 85%): {entities_list}.  
    • Focus on the lead (main point), key developments, and concluding insight—omit minor details.  
    • Maintain a clear, neutral, informative tone—no hype or sensationalism.  
    • Use only facts from the article—do not add or invent information.

    Article:
    {text}

    Summary:
    """

    try:
        # 4. Generate summary via Gemini
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        cfg = GenerationConfig(
            max_output_tokens=max_words * 3,  # token cushion for ~50 words
            temperature=0.2,
            top_p=0.9,
            top_k=40,
        )
        resp = model.generate_content(prompt, generation_config=cfg)
        summary = resp.text.strip()
        logger.info("Raw summary: %s", summary)

        # 5. Optionally trim if the model greatly overshoots
        words = summary.split()
        if len(words) > max_words * 1.5:
            summary = " ".join(words[: max_words * 1])  # keep first ~max_words
            logger.info("Trimmed summary to around %d words", max_words)

        return summary

    except Exception as e:
        logger.error("Gemini error: %s", e)
        raise