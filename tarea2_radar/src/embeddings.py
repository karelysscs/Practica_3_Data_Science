"""Embeddings locales para el RAG híbrido (Fase 3). Mismo modelo y misma lógica
de prefijos query:/passage: que la Tarea 1 (ver esa carpeta para el porqué de
la elección del modelo). Módulo independiente para que la Tarea 2 no dependa
de la Tarea 1 en tiempo de ejecución. No importa UI.
"""
from __future__ import annotations

import numpy as np


class LocalEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._uses_e5_prefixes = "e5" in model_name.lower()
        try:
            self.dimension = self._model.get_embedding_dimension()
        except AttributeError:
            self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str], batch_size: int = 32, is_query: bool = False) -> np.ndarray:
        if self._uses_e5_prefixes:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        vectors = self._model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        )
        return vectors.astype("float32")


def cosine_search(query_vector: np.ndarray, matrix: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    q = query_vector / (np.linalg.norm(query_vector) + 1e-12)
    scores = matrix @ q
    top_k = min(top_k, len(scores))
    idx = np.argpartition(-scores, top_k - 1)[:top_k] if top_k < len(scores) else np.arange(len(scores))
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]
