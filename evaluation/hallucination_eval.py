import logging
from evaluate import load

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
# Load BERTScore metric (runs on CPU)
# -----------------------------------------------------------------------------
# BERTScore’s `device` arg must be a string, e.g. "cpu"  [oai_citation_attribution:3‡Hugging Face](https://huggingface.co/spaces/evaluate-metric/bertscore/blob/main/bertscore.py?utm_source=chatgpt.com)
bertscore = load("bertscore")

def evaluate_hallucination_cpu(article: str, summary: str) -> float:
    """
    Computes BERTScore F1 as a proxy for factual consistency.
    Returns a float (higher = more factually consistent).
    """
    logger.info("Computing BERTScore (CPU)…")
    result = bertscore.compute(
        model_type="microsoft/deberta-xlarge-mnli",
        predictions=[summary],
        references=[article],
        device="cpu",       # enforce CPU, no negative indices  [oai_citation_attribution:4‡Hugging Face](https://huggingface.co/spaces/evaluate-metric/bertscore/blob/main/bertscore.py?utm_source=chatgpt.com)
        batch_size=1
    )
    score = result["f1"][0]
    logger.info("BERTScore F1 result: %s", score)
    return float(score)