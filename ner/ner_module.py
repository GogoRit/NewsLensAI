# evaluation/ner_module.py

from transformers import AutoTokenizer, pipeline
from typing import List, Dict
from evaluation._utils import safe_truncate

MODEL = "dbmdz/bert-large-cased-finetuned-conll03-english"

# load tokenizer & pipeline once
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
ner_pipe   = pipeline(
    "ner",
    model=MODEL,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=-1
)

def extract_entities(article_text: str, summary_text: str) -> Dict[str, List[Dict]]:
    def _run(txt: str):
        safe = safe_truncate(txt, tokenizer)
        ents = ner_pipe(safe)
        return [
            {"text": e["word"], "label": e["entity_group"], "score": round(e["score"], 4)}
            for e in ents
        ]

    return {
        "article_entities": _run(article_text),
        "summary_entities": _run(summary_text)
    }