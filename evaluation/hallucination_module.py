# evaluation/hallucination_module.py

from sentence_transformers import SentenceTransformer
import numpy as np

# Embedding model loads internally (handles its own truncation)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

def compute_similarity(article_text: str, summary_text: str) -> float:
    art_embeds = embedder.encode([article_text], convert_to_numpy=True)
    art_vec    = np.mean(art_embeds, axis=0)
    sum_vec    = embedder.encode([summary_text], convert_to_numpy=True)[0]
    na, nb     = np.linalg.norm(art_vec), np.linalg.norm(sum_vec)
    if na == 0 or nb == 0:
        return 0.0
    cos = float(np.dot(art_vec, sum_vec) / (na * nb))
    return round((cos + 1) / 2, 4)