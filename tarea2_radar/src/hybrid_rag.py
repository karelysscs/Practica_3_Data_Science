"""Fase 3 — Motor de RAG híbrido.

Regla de arquitectura (misma que Tarea 1): un único punto de entrada público
(`answer_question`) que devuelve un resultado estructurado. No importa
Streamlit ni ninguna librería de UI.

"Híbrido" = condiciones NUMÉRICAS (monto mínimo/máximo) y TERRITORIALES
(departamento) se aplican como FILTROS ESTRUCTURADOS sobre la tabla (pandas),
NUNCA metidas dentro del texto a embeber. Los embeddings solo indexan la
descripción semántica del proceso. Esto evita que, por ejemplo, "menos de
S/ 50,000" se intente resolver por similitud semántica (no funcionaría).

Abstención: se reutiliza la misma lógica de la Tarea 1 — se calcula la
similitud ANTES de llamar al LLM y se abstiene sin costo si no alcanza el
umbral calibrado, o si los filtros no dejan ningún proceso candidato.

Citación: por OCID (identificador único de proceso en OCDS), no por página.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.cost_logger import log_cost_entry
from src.embeddings import cosine_search
from src.llm_client import call_llm


@dataclass
class Filters:
    department: str | None = None
    min_amount: float | None = None
    max_amount: float | None = None
    category: str | None = None
    date_from: str | None = None
    date_to: str | None = None


@dataclass
class CitedProcess:
    ocid: str
    buyer_name: str
    department: str | None
    amount: float | None
    currency: str | None
    title: str
    description: str | None
    similarity: float


@dataclass
class HybridRAGResult:
    question: str
    filters: Filters
    n_candidates_after_filters: int
    abstained: bool
    abstain_reason: str | None
    answer: str | None
    cited_processes: list[CitedProcess]
    top_similarity: float
    retrieval_time_seconds: float
    llm_time_seconds: float | None
    cost_usd: float
    llm_model: str | None


def apply_filters(df: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if filters.department:
        mask &= df["buyer_department"] == filters.department.strip().upper()
    if filters.min_amount is not None:
        mask &= df["tender_value_amount"].fillna(0) >= filters.min_amount
    if filters.max_amount is not None:
        mask &= df["tender_value_amount"].fillna(0) <= filters.max_amount
    if filters.category:
        mask &= df["main_category"] == filters.category
    # Comparación solo por fecha (primeros 10 chars, "AAAA-MM-DD"): el campo
    # trae fecha+hora ISO, así que comparar el string completo contra un
    # date_to de solo fecha excluiría incorrectamente todo ese mismo día
    # (cualquier hora > "00:00:00" ya es lexicográficamente "mayor").
    date_only = df["tender_date_published"].fillna("").str.slice(0, 10)
    if filters.date_from:
        mask &= date_only >= filters.date_from
    if filters.date_to:
        mask &= date_only <= filters.date_to
    return df[mask]


_SYSTEM_PROMPT = """Eres un asistente que responde EXCLUSIVAMENTE con base en los procesos de \
contratación pública que se te entregan a continuación (Portal de Contrataciones Abiertas OECE, \
Perú). Reglas estrictas:
1. Si la respuesta no está respaldada por los procesos entregados, dilo con claridad en vez de \
   inventar.
2. Cada proceso que menciones debe ir citado por su OCID exacto, entre corchetes: [OCID: ...].
3. Sé conciso. No inventes montos, departamentos ni proveedores que no estén en los datos.
"""


def _build_context(cited: list[CitedProcess]) -> str:
    parts = []
    for c in cited:
        parts.append(
            f"### Proceso [OCID: {c.ocid}] (similitud={c.similarity:.3f})\n"
            f"Título: {c.title}\nDescripción: {c.description or '(sin descripción)'}\n"
            f"Comprador: {c.buyer_name} ({c.department})\n"
            f"Monto: {c.amount} {c.currency}"
        )
    return "\n\n".join(parts)


def answer_question(
    question: str,
    filters: Filters,
    df: pd.DataFrame,
    vectors: np.ndarray,
    embedder,
    cfg: dict,
    api_key: str | None,
    llm_model: str,
    cost_log_path,
) -> HybridRAGResult:
    rag_cfg = cfg["rag_engine"]
    threshold = rag_cfg["similarity_threshold"]
    top_k = rag_cfg["top_k"]

    filtered = apply_filters(df, filters)
    n_candidates = len(filtered)

    if n_candidates == 0:
        return HybridRAGResult(
            question=question, filters=filters, n_candidates_after_filters=0,
            abstained=True,
            abstain_reason="Ningún proceso cumple los filtros estructurados (departamento/monto/fecha/categoría) indicados.",
            answer=None, cited_processes=[], top_similarity=0.0,
            retrieval_time_seconds=0.0, llm_time_seconds=None, cost_usd=0.0, llm_model=None,
        )

    filtered_vectors = vectors[filtered.index.to_numpy()]

    t0 = time.perf_counter()
    query_vec = embedder.embed([question], is_query=True)[0]
    hits = cosine_search(query_vec, filtered_vectors, top_k=top_k)
    retrieval_time = time.perf_counter() - t0

    top_similarity = hits[0][1] if hits else 0.0

    cited = []
    filtered_reset = filtered.reset_index(drop=True)
    for row_idx, score in hits:
        row = filtered_reset.iloc[row_idx]
        cited.append(CitedProcess(
            ocid=row["ocid"], buyer_name=row["buyer_name"], department=row["buyer_department"],
            amount=row["tender_value_amount"], currency=row["tender_currency"],
            title=row["tender_title"], description=row["tender_description"], similarity=score,
        ))

    if top_similarity < threshold:
        return HybridRAGResult(
            question=question, filters=filters, n_candidates_after_filters=n_candidates,
            abstained=True,
            abstain_reason=(
                f"Similitud máxima ({top_similarity:.3f}) por debajo del umbral calibrado "
                f"({threshold}). {rag_cfg['abstain_message']}"
            ),
            answer=None, cited_processes=cited, top_similarity=top_similarity,
            retrieval_time_seconds=retrieval_time, llm_time_seconds=None, cost_usd=0.0, llm_model=None,
        )

    provider = cfg["rag_engine"].get("llm_provider", "openai")
    if not api_key:
        raise RuntimeError(
            f"Se superó el umbral de similitud pero no hay API key configurada para "
            f"el proveedor '{provider}' en .env."
        )

    context = _build_context(cited)
    user_prompt = f"Pregunta: {question}\n\nProcesos disponibles:\n\n{context}"

    llm_resp = call_llm(provider, llm_model, _SYSTEM_PROMPT, user_prompt, api_key)
    entry = log_cost_entry(
        log_path=cost_log_path, call_type="chat", model=llm_model,
        input_tokens=llm_resp.input_tokens, output_tokens=llm_resp.output_tokens,
        latency_seconds=llm_resp.latency_seconds, pricing_cfg=cfg["pricing"],
        context=f"query ({provider}): {question[:120]}",
    )

    return HybridRAGResult(
        question=question, filters=filters, n_candidates_after_filters=n_candidates,
        abstained=False, abstain_reason=None, answer=llm_resp.text,
        cited_processes=cited, top_similarity=top_similarity,
        retrieval_time_seconds=retrieval_time, llm_time_seconds=llm_resp.latency_seconds,
        cost_usd=entry.usd_cost, llm_model=llm_model,
    )
