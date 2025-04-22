# evaluation/cpu_bias.py

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
# Load BLEURT metric
# -----------------------------------------------------------------------------
# According to the BLEURT Evaluate module, the compute() signature is:
#   compute(predictions: List[str], references: List[str], checkpoint: str = None)
# ’batch_size’ is not supported.  [oai_citation_attribution:0‡GitHub](https://github.com/huggingface/evaluate/blob/main/metrics/bleurt/bleurt.py?utm_source=chatgpt.com) [oai_citation_attribution:1‡Stack Overflow](https://stackoverflow.com/questions/79319294/bleurt-evaluation-metric-consumed-too-much-ram?utm_source=chatgpt.com)
bleurt = load("bleurt")

def evaluate_bias_cpu(text: str) -> float:
    """
    Computes BLEURT on the text itself as a proxy for neutrality (lower value = more extreme bias).
    Returns:
        float: The BLEURT score.
    """
    logger.info("Computing BLEURT for bias proxy…")
    # Remove batch_size; use only predictions, references, and optional checkpoint  [oai_citation_attribution:2‡GitHub](https://github.com/huggingface/evaluate/blob/main/metrics/bleurt/bleurt.py?utm_source=chatgpt.com)
    result = bleurt.compute(
        predictions=[text],
        references=[text],
        # you may specify a smaller checkpoint if desired:
        # checkpoint="bleurt-tiny-128"
    )
    score = result["scores"][0]
    logger.info("BLEURT result: %s", score)
    return float(score)