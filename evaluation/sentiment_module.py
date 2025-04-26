# evaluation/sentiment_module.py

from transformers import AutoTokenizer, pipeline
from typing import Dict
from evaluation._utils import safe_truncate

MODEL      = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENT_MAP   = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

tokenizer        = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
sentiment_pipe   = pipeline(
    "text-classification",
    model=MODEL,
    tokenizer=tokenizer,
    device=-1
)

def analyze_sentiment(article_text: str, summary_text: str) -> Dict[str, Dict]:
    art_safe = safe_truncate(article_text, tokenizer)
    sum_safe = safe_truncate(summary_text, tokenizer)

    art = sentiment_pipe(art_safe)[0]
    sm  = sentiment_pipe(sum_safe)[0]

    return {
        "article_sentiment": {"label": SENT_MAP.get(art["label"], art["label"]), "score": round(art["score"],4)},
        "summary_sentiment": {"label": SENT_MAP.get(sm["label"],  sm["label"]),  "score": round(sm["score"],4)}
    }