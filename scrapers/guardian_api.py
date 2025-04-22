# scrapers/guardian_api.py

import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

GUARDIAN_KEY = os.getenv("GUARDIAN_API_KEY")
if not GUARDIAN_KEY:
    raise RuntimeError("Please set GUARDIAN_API_KEY in your .env")

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

def fetch_guardian_articles(
    query: str,
    page_size: int = 10,
    show_fields: str = "bodyText,headline",
    section: str = None
):
    """
    Fetches articles from The Guardian matching `query`.
    Returns a list of dicts with 'id', 'webTitle', 'webUrl', and 'bodyText'.
    """
    url = "https://content.guardianapis.com/search"
    params = {
        "api-key":     GUARDIAN_KEY,
        "q":           query,
        "page-size":   page_size,
        "show-fields": show_fields,
    }
    if section:
        params["section"] = section

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("response", {})
    articles = []

    for item in data.get("results", []):
        fields = item.get("fields", {})
        body = fields.get("bodyText", "").strip()
        if not body:
            logger.warning("No bodyText for %s", item.get("webUrl"))
        articles.append({
            "id":       item["id"],
            "title":    fields.get("headline", item["webTitle"]),
            "link":     item["webUrl"],
            "full_text": body,
        })

    return articles