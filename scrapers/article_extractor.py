# scrapers/article_extractor.py

import logging
import trafilatura
from newspaper import Article as NewsArticle
from readability import Document
import requests

# -----------------------------------------------------------------------------
# Logger setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def extract_full_text(url: str, timeout: int = 10) -> str:
    """
    Attempts to fetch the *full* article text from `url` via three strategies:
      1. Trafilatura
      2. Newspaper3k
      3. Readability‑lxml

    Returns plain text, or empty string on total failure.
    """
    # 1) Trafilatura
    try:
        html = trafilatura.fetch_url(url, timeout=timeout)
        if html:
            txt = trafilatura.extract(html, include_comments=False, include_tables=False, no_fallback=False)
            if txt and len(txt) > 200:
                logger.info("Trafilatura succeeded for %s", url)
                return txt.strip()
    except Exception as e:
        logger.warning("Trafilatura failed (%s): %s", url, e)

    # 2) Newspaper3k
    try:
        art = NewsArticle(url)
        art.download()
        art.parse()
        txt = art.text
        if txt and len(txt) > 200:
            logger.info("Newspaper3k succeeded for %s", url)
            return txt.strip()
    except Exception as e:
        logger.warning("Newspaper3k failed (%s): %s", url, e)

    # 3) Readability‑lxml
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        doc = Document(resp.text)
        content_html = doc.summary()      # clean HTML fragment
        # strip tags for plain text
        from bs4 import BeautifulSoup
        txt = BeautifulSoup(content_html, "html.parser").get_text(separator="\n").strip()
        if txt and len(txt) > 200:
            logger.info("Readability succeeded for %s", url)
            return txt
    except Exception as e:
        logger.warning("Readability failed (%s): %s", url, e)

    # All strategies failed
    logger.error("All extractors failed for %s", url)
    return ""