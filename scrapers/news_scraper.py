# scrapers/news_scraper.py

import logging
import requests
import feedparser
from typing import List, Dict
from time import sleep

# -----------------------------------------------------------------------------
# Logger setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEFAULT_RSS_URL = "https://news.google.com/rss/search?q=abortion+rights&hl=en-US&gl=US&ceid=US:en"

def get_top_articles(
    rss_url: str = DEFAULT_RSS_URL,
    limit: int = 10,
    timeout: int = 10,
    retries: int = 3,
    backoff: float = 1.0
) -> List[Dict]:
    """
    Fetch the top N articles from an RSS feed, with retries and basic validation.

    Args:
        rss_url (str): URL of the RSS feed.
        limit (int): Max number of articles to return.
        timeout (int): HTTP request timeout (seconds).
        retries (int): Number of retry attempts on failure.
        backoff (float): Base seconds to sleep between retries (multiplied by attempt).

    Returns:
        list of dict: Each dict has keys 'title', 'link', 'published', 'summary'.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    headers = {
        "User-Agent": "NewsLensAI/1.0 (https://github.com/GogoRit/NewsLensAI)"
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(rss_url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)

            if feed.bozo:
                logger.warning("Malformed feed detected (bozo_exception): %s", feed.bozo_exception)

            articles = []
            for entry in feed.entries[:limit]:
                title = getattr(entry, "title", "").strip()
                link = getattr(entry, "link", "").strip()
                # published might be under 'published' or 'updated'
                published = getattr(entry, "published", None) or getattr(entry, "updated", "")
                summary = getattr(entry, "summary", "").strip()

                if not title or not link:
                    logger.warning("Skipping entry due to missing title or link: %r", entry)
                    continue

                articles.append({
                    "title": title,
                    "link": link,
                    "published": published,
                    "summary": summary,
                })

            return articles

        except requests.RequestException as e:
            logger.error("Attempt %d/%d — HTTP error fetching %s: %s", attempt, retries, rss_url, e)
        except Exception as e:
            logger.error("Attempt %d/%d — Unexpected error: %s", attempt, retries, e)

        if attempt < retries:
            sleep(backoff * attempt)
            logger.info("Retrying (attempt %d)...", attempt + 1)
        else:
            raise RuntimeError(f"Failed to fetch feed after {retries} attempts")

    return []  # fallback, though in practice the exception above will fire