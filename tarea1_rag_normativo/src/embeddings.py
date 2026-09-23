"""Fase 2 / Fase 4 — Embeddings locales y de OpenAI, con interfaz común.

No importa UI. Las llamadas a OpenAI registran costo real vía cost_logger
(latencia medida, tokens reportados por la API, tarifa desde config.yaml).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.cost_logger import Timer, log_cost_entry


class LocalEmbedder:
    """Embeddings locales (sentence-transformers), sin costo ni llamadas externas.

    Los modelos de la familia E5 (intfloat/multilingual-e5-*) están entrenados con
    prefijos asimétricos "query: " / "passage: " para consulta vs. fragmento indexado;
    usarlos mejora sustancialmente el retrieval frente a un modelo de similitud
    genérico. Se detecta automáticamente por el nombre del modelo.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # import perezoso: pesado
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
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.astype("float32")


class OpenAIEmbedder:
    """Embeddings vía API de OpenAI (text-embedding-3-small u otro configurado)."""

    def __init__(self, model_name: str, api_key: str, pricing_cfg: dict, cost_log_path: Path):
        from openai import OpenAI  # import perezoso
        self.model_name = model_name
        self._client = OpenAI(api_key=api_key)
        self.pricing_cfg = pricing_cfg
        self.cost_log_path = cost_log_path
        self.dimension = None  # se fija tras la primera llamada real

    def embed(self, texts: list[str], batch_size: int = 64, context: str = "indexación") -> np.ndarray:
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            with Timer() as t:
                resp = self._client.embeddings.create(model=self.model_name, input=batch)
            vectors = np.array([d.embedding for d in resp.data], dtype="float32")
            if self.dimension is None:
                self.dimension = vectors.shape[1]
            all_vectors.append(vectors)

            usage = resp.usage
            log_cost_entry(
                log_path=self.cost_log_path,
                call_type="embedding",
                model=self.model_name,
                input_tokens=usage.total_tokens,
                output_tokens=0,
                latency_seconds=t.elapsed,
                pricing_cfg=self.pricing_cfg,
                context=f"{context} (batch {i // batch_size + 1}, {len(batch)} fragmentos)",
            )
        return np.vstack(all_vectors)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normaliza L2 por fila; usado para embeddings que no vienen ya normalizados
    (los locales de sentence-transformers ya se piden normalizados)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
