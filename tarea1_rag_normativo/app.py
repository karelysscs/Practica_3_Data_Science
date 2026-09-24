"""Fase 5 — Interfaz Streamlit (proceso ONLINE).

Arranque con un solo comando desde un entorno limpio:
    streamlit run app.py

Este archivo es la ÚNICA parte del proyecto que puede importar Streamlit.
Toda la lógica de negocio vive en src/ (engine sin dependencias de UI) y se
consume aquí a través de una sola función pública: src.rag_engine.answer_question.

Carga el índice YA CONSTRUIDO (no reconstruye nada): si no existe, muestra
instrucciones para correr build_index.py primero.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.config import get_llm_credentials, get_openai_api_key, load_config, resolve_path
from src.embeddings import LocalEmbedder, OpenAIEmbedder
from src.index_store import load_existing
from src.rag_engine import answer_question

st.set_page_config(page_title="RAG Normativo — Contrataciones Públicas", page_icon="📜", layout="wide")

cfg = load_config()
processed_dir = resolve_path(cfg["paths"]["processed_dir"])
logs_dir = resolve_path(cfg["paths"]["logs_dir"])
eval_dir = resolve_path(cfg["paths"]["eval_dir"])
selected_config = cfg["chunking"]["selected_config"]


@st.cache_resource(show_spinner="Cargando modelo de embeddings local...")
def _get_local_embedder():
    return LocalEmbedder(cfg["embeddings"]["local"]["model_name"])


def _available_indexes() -> dict[str, Path]:
    candidates = {
        "local": processed_dir / f"index_{selected_config}__local",
        "openai": processed_dir / f"index_{selected_config}__openai",
    }
    return {label: path for label, path in candidates.items() if load_existing(path) is not None}


st.title(cfg["streamlit"]["app_title"])
st.caption(
    "Corpus: Ley N.° 32069 (Ley General de Contrataciones Públicas) y "
    "Decreto Supremo N.° 001-2026-EF. Las respuestas citan documento y página; "
    "el asistente se abstiene si la pregunta no está respaldada por el corpus."
)

tab_query, tab_quality, tab_eval, tab_costs = st.tabs(
    ["🔎 Consulta", "🧪 Calidad de extracción", "📊 Evaluación", "💵 Costos"]
)

# ==========================================================================
# TAB 1 — Consulta
# ==========================================================================
with tab_query:
    available = _available_indexes()
    if not available:
        st.error(
            "No hay ningún índice construido todavía. Corre primero, desde esta carpeta:\n\n"
            "```\npython build_index.py\n```"
        )
    else:
        col_left, col_right = st.columns([3, 1])
        with col_right:
            embedder_label = st.radio(
                "Índice / embeddings a usar",
                options=list(available.keys()),
                format_func=lambda x: "Local (gratis)" if x == "local" else "OpenAI text-embedding-3-small",
            )
            st.caption(f"Umbral de abstención configurado: **{cfg['rag_engine']['similarity_threshold']}**")
            _provider, _model, _ = get_llm_credentials(cfg)
            st.caption(f"LLM de respuesta: **{_provider}** ({_model})")

        with col_left:
            question = st.text_input(
                "Escribe tu pregunta sobre contrataciones públicas (Ley 32069 / DS 001-2026-EF):",
                placeholder="Ej: ¿Hasta cuántas UIT se considera un contrato menor?",
            )
            submitted = st.button("Preguntar", type="primary", disabled=not question)

        if submitted and question:
            index_dir = available[embedder_label]
            if embedder_label == "local":
                embedder = _get_local_embedder()
            else:
                api_key = get_openai_api_key()
                if not api_key:
                    st.error("No hay OPENAI_API_KEY configurada en .env.")
                    st.stop()
                embedder = OpenAIEmbedder(
                    cfg["embeddings"]["openai"]["model_name"], api_key, cfg["pricing"],
                    logs_dir / "cost_log.jsonl",
                )

            llm_provider, llm_model, llm_api_key = get_llm_credentials(cfg)

            with st.spinner("Buscando en el corpus y generando respuesta..."):
                try:
                    result = answer_question(
                        question=question,
                        index_dir=index_dir,
                        embedder=embedder,
                        cfg=cfg,
                        api_key=llm_api_key,
                        llm_model=llm_model,
                        cost_log_path=logs_dir / "cost_log.jsonl",
                    )
                except Exception as exc:
                    st.error(f"Error al generar la respuesta: {exc}")
                    result = None

            if result:
                if result.abstained:
                    st.warning(f"🛑 **El asistente se abstiene de responder.**\n\n{result.abstain_reason}")
                else:
                    st.success(result.answer)

                colA, colB, colC = st.columns(3)
                colA.metric("Similitud top-1", f"{result.top_similarity:.3f}")
                colB.metric("Costo de esta consulta (USD)", f"${result.cost_usd:.6f}")
                colC.metric(
                    "Latencia (retrieval + LLM)",
                    f"{result.retrieval_time_seconds + (result.llm_time_seconds or 0):.2f}s",
                )

                st.markdown("#### Fragmentos citados")
                for frag in result.cited_fragments:
                    page_ref = f"p. {frag.page_start}" if frag.page_start == frag.page_end else f"pp. {frag.page_start}-{frag.page_end}"
                    with st.expander(f"{frag.doc_title} — {page_ref} (similitud {frag.similarity:.3f})"):
                        st.text(frag.text_snippet)

# ==========================================================================
# TAB 2 — Calidad de extracción (Fase 1)
# ==========================================================================
with tab_quality:
    st.markdown("### Reportes de calidad por documento (Fase 1)")
    for doc_cfg in cfg["documents"]:
        report_path = logs_dir / f"quality_report_{doc_cfg['id']}.md"
        if report_path.exists():
            with st.expander(doc_cfg["title"], expanded=False):
                st.markdown(report_path.read_text(encoding="utf-8"))
        else:
            st.info(f"Aún no existe el reporte de {doc_cfg['id']}. Corre `python build_index.py`.")

    chunking_report_path = logs_dir / "phase2_chunking_report.md"
    if chunking_report_path.exists():
        st.markdown("### Comparación de configuraciones de chunking (Fase 2)")
        st.markdown(chunking_report_path.read_text(encoding="utf-8"))

# ==========================================================================
# TAB 3 — Evaluación (Fase 4)
# ==========================================================================
with tab_eval:
    eval_report_path = logs_dir / "phase4_evaluation_report.md"
    results_path = eval_dir / "results_embeddings_comparison.json"
    if eval_report_path.exists():
        st.markdown(eval_report_path.read_text(encoding="utf-8"))
        if results_path.exists():
            with st.expander("Ver JSON completo de resultados (por pregunta)"):
                st.json(json.loads(results_path.read_text(encoding="utf-8")))
    else:
        st.info("Aún no hay resultados de evaluación. Corre `python -m eval.run_eval`.")

# ==========================================================================
# TAB 4 — Costos (log observable, Fase 3)
# ==========================================================================
with tab_costs:
    cost_log_path = logs_dir / "cost_log.jsonl"
    if cost_log_path.exists():
        entries = [json.loads(l) for l in cost_log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        total_cost = sum(e["usd_cost"] for e in entries)
        st.metric("Costo total registrado (USD)", f"${total_cost:.6f}")
        st.caption(f"{len(entries)} llamadas registradas · tarifas de referencia: {cfg['pricing']['reference_date']}")
        st.dataframe(entries, use_container_width=True)
    else:
        st.info("Todavía no se registró ninguna llamada con costo.")
