"""Fase 4 — Dashboard Streamlit (proceso ONLINE). Único archivo que importa
Streamlit; toda la lógica vive en src/ (sin dependencias de UI).

Arranque con un solo comando desde un entorno limpio:
    streamlit run app.py

Lee los archivos YA PROCESADOS (no reconstruye nada): si no existen, muestra
instrucciones para correr build_data.py primero.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import get_openai_api_key, get_openai_chat_model, load_config, resolve_path
from src.embeddings import LocalEmbedder
from src.hybrid_rag import Filters, answer_question, apply_filters
from src.index_store import load_index
from src.risk_indicator import single_bidder_share_by_department, top_buyers_by_single_bidder_share

st.set_page_config(page_title="Radar de Contrataciones Públicas", page_icon="🗺️", layout="wide")

cfg = load_config()
processed_dir = resolve_path(cfg["paths"]["processed_dir"])
logs_dir = resolve_path(cfg["paths"]["logs_dir"])
eval_dir = resolve_path(cfg["paths"]["eval_dir"])
assets_dir = Path(__file__).parent / "assets"


@st.cache_data(show_spinner="Cargando dataset de procesos...")
def _load_df() -> pd.DataFrame:
    return pd.read_parquet(processed_dir / "processes.parquet")


@st.cache_data(show_spinner=False)
def _load_index():
    return load_index(processed_dir / "index_local")


@st.cache_resource(show_spinner="Cargando modelo de embeddings local...")
def _get_embedder():
    return LocalEmbedder(cfg["embeddings"]["local"]["model_name"])


@st.cache_data(show_spinner=False)
def _load_geojson():
    with open(assets_dir / "peru_departamentos.geojson", encoding="utf-8") as f:
        return json.load(f)


processed_path = processed_dir / "processes.parquet"
if not processed_path.exists():
    st.error("No hay datos procesados todavía. Corre primero, desde esta carpeta:\n\n```\npython build_data.py\n```")
    st.stop()

df = _load_df()
vectors, manifest = _load_index()

st.title(cfg["streamlit"]["app_title"])
st.caption(
    "Fuente: Portal de Contrataciones Abiertas del OECE (SEACE V3), estándar OCDS. "
    f"{manifest['row_count']} procesos indexados (jul-sep 2026)."
)

# ==========================================================================
# Sidebar — filtros globales
# ==========================================================================
st.sidebar.header("Filtros")
departments = ["(Todos)"] + sorted(df.loc[df["department_is_valid"], "buyer_department"].dropna().unique().tolist())
sel_department = st.sidebar.selectbox("Departamento", departments)
categories = ["(Todas)"] + sorted(df["main_category"].dropna().unique().tolist())
sel_category = st.sidebar.selectbox("Categoría", categories)
amount_max_data = float(df["tender_value_amount"].fillna(0).quantile(0.99))
sel_amount_range = st.sidebar.slider("Rango de monto (S/)", 0.0, max(amount_max_data, 1.0), (0.0, amount_max_data))
dates = df["tender_date_published"].dropna()
if len(dates):
    min_date, max_date = dates.min()[:10], dates.max()[:10]
    st.sidebar.caption(f"Rango de fechas disponible: {min_date} a {max_date}")
sel_date_from = st.sidebar.text_input("Fecha desde (AAAA-MM-DD, opcional)", "")
sel_date_to = st.sidebar.text_input("Fecha hasta (AAAA-MM-DD, opcional)", "")

filters = Filters(
    department=None if sel_department == "(Todos)" else sel_department,
    category=None if sel_category == "(Todas)" else sel_category,
    min_amount=sel_amount_range[0] if sel_amount_range[0] > 0 else None,
    max_amount=sel_amount_range[1] if sel_amount_range[1] < amount_max_data else None,
    date_from=sel_date_from or None,
    date_to=sel_date_to or None,
)

filtered_df = apply_filters(df, filters)

tab_map, tab_query, tab_table, tab_dist, tab_quality, tab_risk = st.tabs(
    ["🗺️ Mapa y KPIs", "🔎 Pregunta (RAG híbrido)", "📋 Tabla / CSV", "📊 Distribuciones", "🧪 Calidad de datos", "⚠️ Riesgo"]
)

# ==========================================================================
# TAB — Mapa y KPIs
# ==========================================================================
with tab_map:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Procesos (filtrados)", f"{len(filtered_df):,}")
    total_amount = filtered_df["tender_value_amount"].fillna(0).sum()
    c2.metric("Monto total convocado (S/)", f"{total_amount:,.0f}")
    c3.metric("Departamentos", filtered_df.loc[filtered_df["department_is_valid"], "buyer_department"].nunique())
    c4.metric("Compradores únicos", filtered_df["buyer_id"].nunique())

    by_dept = (
        filtered_df[filtered_df["department_is_valid"]]
        .groupby("buyer_department")
        .agg(n_procesos=("ocid", "count"), monto_total=("tender_value_amount", lambda s: s.fillna(0).sum()))
        .reset_index()
    )
    if len(by_dept):
        geojson = _load_geojson()
        fig = px.choropleth(
            by_dept, geojson=geojson, locations="buyer_department", featureidkey="properties.NOMBDEP",
            color="n_procesos", color_continuous_scale="Blues",
            hover_data={"monto_total": ":,.0f"},
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay procesos con departamento válido para los filtros seleccionados.")

# ==========================================================================
# TAB — Pregunta (RAG híbrido)
# ==========================================================================
with tab_query:
    st.caption(f"Los filtros del panel izquierdo se aplican también a esta consulta. "
               f"Umbral de abstención: {cfg['rag_engine']['similarity_threshold']}")
    question = st.text_input("Pregunta sobre los procesos de contratación:",
                              placeholder="Ej: ¿Hay procesos de compra de ambulancias?")
    ask = st.button("Preguntar", type="primary", disabled=not question)

    if ask and question:
        embedder = _get_embedder()
        api_key = get_openai_api_key()
        llm_model = get_openai_chat_model(cfg)
        with st.spinner("Buscando procesos relevantes..."):
            try:
                result = answer_question(
                    question=question, filters=filters, df=df, vectors=vectors, embedder=embedder,
                    cfg=cfg, api_key=api_key, llm_model=llm_model, cost_log_path=logs_dir / "cost_log.jsonl",
                )
            except Exception as exc:
                st.error(f"Error al generar la respuesta: {exc}")
                result = None

        if result:
            st.caption(f"Candidatos tras aplicar filtros: {result.n_candidates_after_filters}")
            if result.abstained:
                st.warning(f"🛑 **El asistente se abstiene.**\n\n{result.abstain_reason}")
            else:
                st.success(result.answer)
                st.metric("Costo de esta consulta (USD)", f"${result.cost_usd:.6f}")

            st.markdown("#### Procesos citados (por OCID)")
            for p in result.cited_processes:
                with st.expander(f"[{p.ocid}] {p.buyer_name} — {p.department} (similitud {p.similarity:.3f})"):
                    st.write(f"**Título:** {p.title}")
                    st.write(f"**Monto:** {p.amount} {p.currency}")

# ==========================================================================
# TAB — Tabla / CSV
# ==========================================================================
with tab_table:
    sort_col = st.selectbox("Ordenar por", ["tender_value_amount", "tender_date_published", "num_tenderers"])
    sorted_df = filtered_df.sort_values(sort_col, ascending=False)
    display_cols = ["ocid", "buyer_name", "buyer_department", "tender_title", "main_category",
                     "tender_value_amount", "tender_currency", "tender_date_published", "num_tenderers"]
    st.dataframe(sorted_df[display_cols], use_container_width=True, height=450)
    st.download_button(
        "⬇️ Descargar CSV (filtrado)", sorted_df[display_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="procesos_filtrados.csv", mime="text/csv",
    )

# ==========================================================================
# TAB — Distribuciones
# ==========================================================================
with tab_dist:
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(filtered_df[filtered_df["tender_value_amount"] > 0], x="tender_value_amount",
                             nbins=40, title="Distribución de montos (> S/ 0)")
        st.plotly_chart(fig1, use_container_width=True)
        fig3 = px.bar(filtered_df["main_category"].value_counts().reset_index(),
                      x="main_category", y="count", title="Procesos por categoría")
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        fig2 = px.bar(filtered_df["buyer_department"].value_counts().head(15).reset_index(),
                      x="buyer_department", y="count", title="Top 15 departamentos por N° de procesos")
        st.plotly_chart(fig2, use_container_width=True)
        fig4 = px.bar(filtered_df["source_month"].value_counts().sort_index().reset_index(),
                      x="source_month", y="count", title="Procesos por mes de origen")
        st.plotly_chart(fig4, use_container_width=True)

# ==========================================================================
# TAB — Calidad de datos
# ==========================================================================
with tab_quality:
    quality_report_path = logs_dir / "quality_report.md"
    if quality_report_path.exists():
        st.markdown(quality_report_path.read_text(encoding="utf-8"))
    else:
        st.info("Aún no existe el reporte de calidad. Corre `python build_data.py`.")

    eval_report_path = logs_dir / "phase4_evaluation_report.md"
    if eval_report_path.exists():
        st.markdown("---")
        st.markdown(eval_report_path.read_text(encoding="utf-8"))

# ==========================================================================
# TAB — Riesgo (Fase 5)
# ==========================================================================
with tab_risk:
    st.markdown(
        "Proporción de adjudicaciones con **un solo postor** (single-bidder awards), "
        "una de las señales de riesgo de integridad más citadas por la Open Contracting "
        "Partnership (OCP) en su literatura sobre banderas rojas de contrataciones públicas."
    )
    by_dept_risk = single_bidder_share_by_department(df)
    st.markdown("#### Por departamento")
    st.dataframe(by_dept_risk, use_container_width=True)

    st.markdown(f"#### Top {cfg['risk']['top_n_buyers']} compradores "
                f"(mínimo {cfg['risk']['min_processes_for_ranking']} adjudicaciones)")
    top_buyers = top_buyers_by_single_bidder_share(
        df, cfg["risk"]["min_processes_for_ranking"], cfg["risk"]["top_n_buyers"]
    )
    st.dataframe(top_buyers, use_container_width=True)
