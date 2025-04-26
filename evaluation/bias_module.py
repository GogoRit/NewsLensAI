# evaluation/bias_module.py

from transformers import AutoTokenizer, pipeline
from typing import Dict
from evaluation._utils import safe_truncate

MODEL    = "bucketresearch/politicalBiasBERT"
LABEL_MAP = {"LABEL_0": "Left", "LABEL_1": "Center", "LABEL_2": "Right"}

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
bias_pipe = pipeline(
    "text-classification",
    model=MODEL,
    tokenizer=tokenizer,
    device=-1
)

def detect_bias(article_text: str, summary_text: str) -> Dict[str, Dict]:
    art_safe = safe_truncate(article_text, tokenizer)
    sum_safe = safe_truncate(summary_text, tokenizer)

    art = bias_pipe(art_safe)[0]
    sm  = bias_pipe(sum_safe)[0]

    return {
        "article_bias": {"label": LABEL_MAP.get(art["label"], art["label"]), "score": round(art["score"],4)},
        "summary_bias": {"label": LABEL_MAP.get(sm["label"],  sm["label"]),  "score": round(sm["score"],4)}
    }