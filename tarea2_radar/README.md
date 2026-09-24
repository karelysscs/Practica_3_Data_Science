# Tarea 2 — Radar de Contrataciones Públicas

Dashboard con RAG híbrido sobre datos reales del Portal de Contrataciones
Abiertas del OECE (Perú), estándar OCDS.

## Cómo correrla

Desde esta carpeta (`tarea2_radar/`):

```bash
python build_data.py                # Fases 1-3: adquisición, validación, índice
python build_data.py --incremental  # además, trae actualizaciones recientes vía API
python -m eval.run_eval             # Fase 4: evaluación (Recall@k, calibración)
streamlit run app.py                # Fase 4/5: dashboard
```

`build_data.py` es idempotente: los archivos mensuales no se re-descargan si ya
existen; el índice de embeddings se construye por lotes y es **resumible**
(si se interrumpe, retoma desde el último lote guardado en vez de re-embeber
todo — necesario en la práctica: con ~18,000 procesos, en una laptop bajo
carga el embebido puede tardar 20-30 minutos).

## Arquitectura

```
tarea2_radar/
├── config.yaml            # única fuente de parámetros
├── build_data.py          # proceso OFFLINE (Fases 1-3)
├── app.py                 # proceso ONLINE / dashboard — único archivo con Streamlit
├── assets/
│   └── peru_departamentos.geojson   # límites departamentales para el mapa (juaneladio/peru-geojson)
├── src/
│   ├── acquisition.py     # Fase 1: descarga masiva + actualizaciones recientes vía API
│   ├── ocds_parse.py      # release vs. record -> 1 fila por proceso, dedup por OCID
│   ├── departments.py     # normalización a los 25 departamentos
│   ├── validation.py      # Fase 2: reporte de calidad
│   ├── embeddings.py      # embeddings locales (mismo modelo que Tarea 1)
│   ├── index_store.py     # índice resumible por lotes
│   ├── hybrid_rag.py      # Fase 3: filtros estructurados + embeddings + abstención + citación por OCID
│   ├── risk_indicator.py  # Fase 5: single-bidder award share
│   └── run_eval.py        # Fase 4
├── eval/
│   ├── eval_set.json      # 13 preguntas (10 in-scope + 3 out-of-scope) con OCID relevantes reales
│   └── results.json
├── data/
│   ├── raw/                # JSON crudo de OECE (NO versionado, ~200MB; se regenera con build_data.py)
│   └── processed/          # processes.parquet/csv + índice (SÍ versionado, ~40MB)
└── logs/                   # reporte de calidad, evaluación, costos
```

## Fase 1 — Adquisición

**Fuente:** [contratacionesabiertas.oece.gob.pe](https://contratacionesabiertas.oece.gob.pe)
(Portal de Contrataciones Abiertas, estándar OCDS).

- **Descarga masiva:** 3 meses de 2026 (julio, agosto, septiembre), formato JSON
  (`/api/v1/file/seace_v3/json/{año}/{mes}`). Re-ejecutable: si el JSON local ya
  existe, no se vuelve a descargar.
- **Actualizaciones recientes vía API (no vía bulk):** `/releases?dateFrom=` para
  descubrir qué OCID tuvieron actividad en los últimos N días (granularidad
  *release*), y por cada uno `/record/{ocid}` para traer su estado compilado
  vigente (granularidad *record*). Con throttling (pausa entre llamadas) y
  cache en disco con TTL para no repetir llamadas innecesarias.
- **Release vs. record (OCDS):** cada archivo bulk trae, por OCID, un
  `compiledRelease` (la vista consolidada vigente) + la lista `releases` que la
  componen (cada una con su fecha y tag). Un mismo proceso real observado tenía
  2 releases (`planning`+`tender`) fusionadas en 1 record — ver
  `src/ocds_parse.py` para el detalle documentado.
- **Una fila por proceso:** 17,915 registros antes de deduplicar → **17,915**
  después de deduplicar por OCID (0 duplicados en este corpus, porque la
  segmentación mensual de OECE ya asigna cada proceso a un único mes por fecha
  de convocatoria).

## Fase 2 — Validación y normalización

Ver [logs/quality_report.md](logs/quality_report.md) completo. Resumen real:

- Monto de convocatoria ausente o en cero: **4,790 / 17,915 (26.7%)** — normal
  en SEACE: muchos procesos se publican antes de tener presupuesto asignado.
- Descripciones ausentes: **0**.
- Departamento del comprador ausente o inválido tras normalizar: **0 / 0**
  (los 25 departamentos ya venían limpios en la fuente; igual se implementó y
  documentó la normalización con alias, ver `src/departments.py`).
- Texto con problema de codificación reparado automáticamente: **0**;
  **2 registros** quedaron marcados como "codificación sospechosa sin reparar"
  porque un fix ciego habría roto acentos correctos en el resto del mismo
  string (ver el reporte para el ejemplo real).

## Fase 3 — RAG híbrido

- Embeddings locales (`intfloat/multilingual-e5-small`, igual que Tarea 1)
  **solo** sobre título+descripción+comprador+departamento+categoría.
- Monto y departamento son **filtros estructurados** sobre la tabla (pandas),
  nunca metidos al texto embebido — así "más de S/ 200,000" se resuelve
  exacto, no por semejanza semántica.
- Abstención antes de llamar al LLM si la similitud no alcanza el umbral
  calibrado (reutiliza la misma arquitectura de la Tarea 1).
- Citación por OCID (identificador único de proceso en OCDS).

## Fase 4 — Evaluación

13 preguntas reales (10 in-scope con OCID relevantes conocidos + 3 out-of-scope),
ver [eval/eval_set.json](eval/eval_set.json) y
[logs/phase4_evaluation_report.md](logs/phase4_evaluation_report.md):

| k | Recall@k |
|---|---|
| 1 | 0.4 |
| 3 | 0.6 |
| 5 | 0.6 |
| 8 | 0.6 |

Umbral de abstención calibrado: **0.844** (similitud in-scope mínima 0.832,
out-of-scope máxima 0.857 — solapamiento leve por una pregunta-trampa
"de dominio pero fuera de corpus", igual patrón que en la Tarea 1).

La comparación con `text-embedding-3-small` de OpenAI y las respuestas reales
del LLM quedaron pendientes de ejecutar por falta de crédito en la cuenta de
OpenAI usada (ver README raíz del repo) — el código está completo
(`src/hybrid_rag.py`) y se verificó que la ruta de abstención (sin costo) y la
ruta de llamada al LLM (hasta el punto de la llamada real) funcionan.

## Fase 5 — Indicador de riesgo

Proporción de adjudicaciones con un solo postor, por departamento y por
comprador (mínimo 5 adjudicaciones para entrar al ranking — evita que un
comprador con 1-2 procesos distorsione el ratio). Referencia: literatura de la
Open Contracting Partnership sobre banderas rojas de integridad en
contrataciones públicas. Hallazgo real: **LIMA concentra 34.9%** de
adjudicaciones a un solo postor; varias entidades específicas superan el 85-100%
(ver pestaña "⚠️ Riesgo" del dashboard o `src/risk_indicator.py`).
