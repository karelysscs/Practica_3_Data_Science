"""Fase 3 — Motor RAG.

Regla de arquitectura: este módulo es el ÚNICO punto de entrada del RAG y
expone una sola función pública (`answer_question`) que devuelve un resultado
estructurado (RAGResult). No importa Streamlit ni ninguna librería de UI:
la interfaz (Fase 5) solo consume esta función.

Lógica de abstención: se calcula la similitud de la pregunta contra el índice
ANTES de llamar al LLM. Si la mejor similitud no alcanza el umbral calibrado
(config.yaml: rag_engine.similarity_threshold, ver eval/results de la Fase 4),
se abstiene sin gastar ni un token de LLM.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.cost_logger import log_cost_entry
from src.index_store import cosine_search, load_existing
from src.llm_client import call_llm


@dataclass
class CitedFragment:
    fragment_id: str
    doc_id: str
    doc_title: str
    version_label: str
    page_start: int
    page_end: int
    similarity: float
    text_snippet: str


@dataclass
class RAGResult:
    question: str
    abstained: bool
    abstain_reason: str | None
    answer: str | None
    cited_fragments: list[CitedFragment]
    top_similarity: float
    retrieval_time_seconds: float
    llm_time_seconds: float | None
    cost_usd: float
    cost_input_tokens: int
    cost_output_tokens: int
    llm_model: str | None


_SYSTEM_PROMPT = """Eres un asistente que responde EXCLUSIVAMENTE con base en los fragmentos \
normativos que se te entregan a continuación (Ley N.° 32069 y/o Decreto Supremo N.° 001-2026-EF). \
Reglas estrictas:
1. Si la respuesta no está explícitamente respaldada por los fragmentos entregados, dilo con \
   claridad ("no encuentro esto en los fragmentos citados") en vez de inventar o usar conocimiento \
   externo sobre contrataciones públicas.
2. Cada afirmación debe ir acompañada de una cita entre corchetes con el formato \
   [documento, p. X] o [documento, pp. X-Y], usando exactamente el nombre de documento indicado \
   en cada fragmento.
3. Si dos fragmentos de documentos distintos comparten el mismo número de artículo, NO los mezcles: \
   son artículos distintos de instrumentos legales distintos. Cita solo el documento correcto.
4. Sé conciso y preciso; no agregues secciones ni interpretaciones no solicitadas.
"""


def _build_context(fragments: list[CitedFragment]) -> str:
    parts = []
    for f in fragments:
        page_ref = f"p. {f.page_start}" if f.page_start == f.page_end else f"pp. {f.page_start}-{f.page_end}"
        parts.append(
            f"### Fragmento — {f.doc_title} ({page_ref}) [similitud={f.similarity:.3f}]\n{f.text_snippet}"
        )
    return "\n\n".join(parts)


def answer_question(
    question: str,
    index_dir: Path,
    embedder,
    cfg: dict,
    api_key: str | None,
    llm_model: str,
    cost_log_path: Path,
) -> RAGResult:
    """Punto de entrada único del motor RAG. `embedder` debe ser el MISMO tipo
    (local u OpenAI) usado para construir el índice en `index_dir`."""
    rag_cfg = cfg["rag_engine"]
    threshold = rag_cfg["similarity_threshold"]
    top_k = rag_cfg["top_k"]
    max_ctx = rag_cfg["max_context_fragments"]

    existing = load_existing(index_dir)
    if existing is None:
        raise FileNotFoundError(f"No existe índice construido en {index_dir}. Ejecuta el build de índice primero.")
    manifest, vectors, fragments_meta = existing

    is_openai = "OpenAI" in type(embedder).__name__
    embed_kwargs = {"context": "rag_query"} if is_openai else {"is_query": True}
    t0 = time.perf_counter()
    query_vec = embedder.embed([question], **embed_kwargs)[0]
    hits = cosine_search(query_vec, vectors, top_k=top_k)
    retrieval_time = time.perf_counter() - t0

    top_similarity = hits[0][1] if hits else 0.0

    cited = []
    for row_idx, score in hits[:max_ctx]:
        frag = fragments_meta[row_idx]
        cited.append(CitedFragment(
            fragment_id=frag.fragment_id,
            doc_id=frag.doc_id,
            doc_title=frag.doc_title,
            version_label=frag.version_label,
            page_start=frag.page_start,
            page_end=frag.page_end,
            similarity=score,
            text_snippet=frag.text,
        ))

    # --- Abstención ANTES de llamar al LLM (requisito de Fase 3) ---
    if top_similarity < threshold:
        return RAGResult(
            question=question,
            abstained=True,
            abstain_reason=(
                f"Similitud máxima ({top_similarity:.3f}) por debajo del umbral calibrado "
                f"({threshold}). {rag_cfg['abstain_message']}"
            ),
            answer=None,
            cited_fragments=cited,
            top_similarity=top_similarity,
            retrieval_time_seconds=retrieval_time,
            llm_time_seconds=None,
            cost_usd=0.0,
            cost_input_tokens=0,
            cost_output_tokens=0,
            llm_model=None,
        )

    provider = cfg["rag_engine"].get("llm_provider", "openai")
    if not api_key:
        raise RuntimeError(
            f"Se superó el umbral de similitud pero no hay API key configurada para "
            f"el proveedor '{provider}' en .env; no se puede llamar al LLM."
        )

    context = _build_context(cited)
    user_prompt = f"Pregunta: {question}\n\nFragmentos disponibles:\n\n{context}"

    llm_resp = call_llm(provider, llm_model, _SYSTEM_PROMPT, user_prompt, api_key)
    answer_text = llm_resp.text

    entry = log_cost_entry(
        log_path=cost_log_path,
        call_type="chat",
        model=llm_model,
        input_tokens=llm_resp.input_tokens,
        output_tokens=llm_resp.output_tokens,
        latency_seconds=llm_resp.latency_seconds,
        pricing_cfg=cfg["pricing"],
        context=f"query ({provider}): {question[:120]}",
    )
    llm_time = llm_resp.latency_seconds

    return RAGResult(
        question=question,
        abstained=False,
        abstain_reason=None,
        answer=answer_text,
        cited_fragments=cited,
        top_similarity=top_similarity,
        retrieval_time_seconds=retrieval_time,
        llm_time_seconds=llm_time,
        cost_usd=entry.usd_cost,
        cost_input_tokens=entry.input_tokens,
        cost_output_tokens=entry.output_tokens,
        llm_model=llm_model,
    )
