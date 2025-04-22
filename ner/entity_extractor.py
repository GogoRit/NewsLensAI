# ner/entity_extractor.py

import logging
from dataclasses import dataclass
from typing import List, Set, Tuple
from transformers import pipeline, Pipeline

from transformers.utils import logging as hf_logging
# ── Suppress deprecation/unmatched‐weights warnings ────────────────
hf_logging.set_verbosity_error()  # only show ERROR+ messages  [oai_citation_attribution:0‡Stack Overflow](https://stackoverflow.com/questions/73221277/python-hugging-face-warning?utm_source=chatgpt.com) [oai_citation_attribution:1‡Hugging Face](https://huggingface.co/docs/transformers/v4.34.0/main_classes/logging?utm_source=chatgpt.com)

import logging
from dataclasses import dataclass

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

# -----------------------------------------------------------------------------
# Entity dataclass
# -----------------------------------------------------------------------------
@dataclass
class Entity:
    text: str
    label: str
    score: float

# -----------------------------------------------------------------------------
# Initialize the NER pipeline once (grouped_entities=True merges subword tokens)
# -----------------------------------------------------------------------------
def _load_ner_pipeline(model_name: str = "dslim/bert-base-NER"):
    """
    Load a Hugging Face NER pipeline using the new aggregation_strategy API
    to replace the deprecated grouped_entities parameter.
    """
    try:
        ner = pipeline(
            task="ner",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple",  # replaces grouped_entities=True
        )
        return ner
    except Exception as e:
        logger.error(f"Failed to load NER pipeline ({model_name}): {e}")
        raise

_NER_PIPELINE = _load_ner_pipeline()


# -----------------------------------------------------------------------------
# Helper: chunk long text into ~max_chars chunks by sentences
# -----------------------------------------------------------------------------
def _chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    sentences = text.split(". ")
    chunks = []
    current = ""
    for sent in sentences:
        # add the sentence plus the period back (except last)
        fragment = sent + (". " if not sent.endswith(".") else "")
        if len(current) + len(fragment) <= max_chars:
            current += fragment
        else:
            if current:
                chunks.append(current.strip())
            current = fragment
    if current:
        chunks.append(current.strip())
    return chunks


# -----------------------------------------------------------------------------
# Main extraction function
# -----------------------------------------------------------------------------
def extract_entities(
    text: str,
    min_score: float = 0.0,
    max_chunk_chars: int = 1000
) -> List[Entity]:
    """
    Extract named entities from `text`, filtering by confidence score.

    Args:
        text: Input document string.
        min_score: Minimum entity confidence to include (0.0–1.0).
        max_chunk_chars: Approximate maximum characters per chunk to avoid OOM.

    Returns:
        List[Entity]: Deduplicated entities in order of appearance.
    """
    if not text or not text.strip():
        return []

    seen: Set[Tuple[str, str]] = set()
    entities: List[Entity] = []
    chunks = _chunk_text(text, max_chars=max_chunk_chars)

    for chunk in chunks:
        try:
            results = _NER_PIPELINE(chunk)
        except Exception as e:
            logger.error(f"NER pipeline failed on chunk: {e}")
            continue

        for res in results:
            ent_text = res.get("word", "").strip()
            ent_label = res.get("entity_group") or res.get("entity") or ""
            ent_score = float(res.get("score", 0.0))

            if not ent_text or ent_score < min_score:
                continue

            key = (ent_text.lower(), ent_label)
            if key in seen:
                continue  # dedupe repeated entity
            seen.add(key)

            entities.append(Entity(text=ent_text, label=ent_label, score=ent_score))

    logger.info(f"Extracted {len(entities)} entities (min_score={min_score})")
    return entities