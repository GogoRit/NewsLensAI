# evaluation/toxicity_module.py

from transformers import AutoTokenizer, pipeline
from typing import Dict
from evaluation._utils import safe_truncate

MODEL     = "JungleLee/bert-toxic-comment-classification"
TOXIC_MAP = {"LABEL_0": "Non-toxic", "LABEL_1": "Toxic"}

tokenizer      = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
toxicity_pipe  = pipeline(
    "text-classification",
    model=MODEL,
    tokenizer=tokenizer,
    device=-1
)

def detect_toxicity(article_text: str, summary_text: str) -> Dict[str, Dict]:
    art_safe = safe_truncate(article_text, tokenizer)
    sum_safe = safe_truncate(summary_text, tokenizer)

    art = toxicity_pipe(art_safe)[0]
    sm  = toxicity_pipe(sum_safe)[0]

    return {
        "article_toxicity": {"label": TOXIC_MAP.get(art["label"], art["label"]), "score": round(art["score"],4)},
        "summary_toxicity": {"label": TOXIC_MAP.get(sm["label"],  sm["label"]),  "score": round(sm["score"],4)}
    }