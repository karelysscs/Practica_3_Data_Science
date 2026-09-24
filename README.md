# Proyecto Integrador — RAG Normativo y Radar de Contrataciones Públicas

Trabajo para el issue [HW_03_202602](https://github.com/d2cml-ai/Data-Science-Python/issues/187):
dos tareas conectadas de Retrieval-Augmented Generation aplicadas a contrataciones
públicas del Perú.

- **Tarea 1** ([tarea1_rag_normativo/](tarea1_rag_normativo/)): asistente RAG sobre normativa
  de contrataciones públicas (Ley N.° 32069 y DS N.° 001-2026-EF), con Streamlit local.
- **Tarea 2** ([tarea2_radar/](tarea2_radar/)): dashboard de datos de contrataciones del Estado
  (17,915 procesos reales, jul-sep 2026) con RAG híbrido (embeddings + filtros
  estructurados), mapa por departamento e indicador de riesgo.

## Estado actual

✅ Tarea 1 completa (Fases 1-5).
✅ Tarea 2 completa (Fases 1-5) — ver [tarea2_radar/README.md](tarea2_radar/README.md).
⏳ Falta: video de presentación.

## Requisitos previos (Windows)

1. Python 3.11+ (probado con 3.14).
2. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```
3. Copiar `.env.example` a `.env` y completar tu clave de OpenAI:

   ```bash
   copy .env.example .env
   ```
   Edita `.env` y pega tu key en `OPENAI_API_KEY=` (ver [platform.openai.com/api-keys](https://platform.openai.com/api-keys)).
   **La cuenta necesita crédito cargado** (mínimo unos USD 5) para que el motor RAG
   (Fase 3) y la comparación con `text-embedding-3-small` (Fase 4) funcionen; sin
   crédito, esas llamadas devuelven `insufficient_quota` (error 429).

## Tarea 1 — Cómo correrla

Desde la carpeta `tarea1_rag_normativo/`:

```bash
cd tarea1_rag_normativo
python build_index.py        # Fases 1 y 2: extracción, limpieza, chunking, índice local
python -m eval.run_eval      # Fase 4: evaluación y comparación local vs. OpenAI
streamlit run app.py         # Fase 5: interfaz (usa el índice ya construido, no reconstruye)
```

`build_index.py` es idempotente: se puede volver a correr sin duplicar fragmentos
ni recomputar embeddings ya calculados. `eval/run_eval.py` construye además el
índice con `text-embedding-3-small` si hay `OPENAI_API_KEY` con crédito disponible.

### Arquitectura

```
tarea1_rag_normativo/
├── config.yaml              # única fuente de parámetros (nada hardcodeado en el código)
├── build_index.py           # proceso OFFLINE (Fases 1-2)
├── app.py                   # proceso ONLINE / UI (Fase 5) — único archivo que importa Streamlit
├── src/
│   ├── config.py            # carga config.yaml y .env
│   ├── pdf_extraction.py    # Fase 1: verificación de fuente + extracción por página
│   ├── cleaning.py          # Fase 1: reglas de limpieza documentadas
│   ├── run_phase1.py        # orquestador Fase 1 + reportes de calidad
│   ├── chunking.py          # Fase 2: fragmentos con metadata (doc/versión/página) e ID estable
│   ├── embeddings.py        # Fase 2/4: embeddings locales (e5) y OpenAI
│   ├── index_store.py       # Fase 2: índice idempotente/resumible + búsqueda coseno
│   ├── run_phase2.py        # orquestador Fase 2 (2 configuraciones de chunking)
│   ├── cost_logger.py       # cálculo y log observable de costo por llamada
│   ├── rag_engine.py        # Fase 3: ÚNICA función pública del motor RAG (sin UI)
│   └── run_phase4_eval.py   # Fase 4: Recall@k, costo, latencia, calibración de umbral
├── eval/
│   ├── eval_set.json        # 20 preguntas (15 in-domain + 5 out-of-domain)
│   ├── run_eval.py          # entrypoint de la Fase 4
│   └── results_embeddings_comparison.json
├── data/
│   ├── raw/                 # PDFs fuente (no versionados, ver Fuentes)
│   └── processed/           # fragmentos, índices (.npy), generados por build_index.py
└── logs/                    # reportes de calidad, comparación de chunking, evaluación, costos
```

**Reglas de arquitectura respetadas:** el proceso offline (`build_index.py`) y el
online (`app.py`) están separados; `src/rag_engine.py` expone una sola función
pública (`answer_question`) que devuelve un resultado estructurado; ningún módulo
de `src/` importa Streamlit; todo parámetro operativo vive en `config.yaml` y las
credenciales solo en `.env` (nunca en el repo, ver `.gitignore`).

### Fuentes (Fase 1)

| Documento | Fuente oficial | Rol |
|---|---|---|
| Ley N.° 32069, Ley General de Contrataciones Públicas (con modificaciones al 19-07-2026) | [gob.pe / colección OECE](https://www.gob.pe/institucion/oece/colecciones/45029-ley-n-32069-ley-general-de-contrataciones-publicas-y-su-reglamento) | Ley original (consolidada) |
| Decreto Supremo N.° 001-2026-EF | [El Peruano, Normas Legales, 08/01/2026](https://busquedas.elperuano.pe/dispositivo/NL/2474920-3) | Modifica el Reglamento de la Ley 32069 |

Ambos son PDF de texto nativo (no escaneados): 0 imágenes embebidas, 0 páginas
sospechosas. Ver `logs/quality_report_<doc_id>.md` para el detalle completo con
ejemplos de limpieza antes/después.

⚠️ **Nota de versiones:** el DS N.° 001-2026-EF modifica el *Reglamento* de la Ley
32069 (no está en este corpus, solo la Ley), y por coincidencia numera sus artículos
igual que el Reglamento (p.ej. "Artículo 15"), que NO corresponde al artículo 15 de
la Ley (temas distintos). El set de evaluación (`eval/eval_set.json`) incluye
preguntas diseñadas específicamente para verificar que el motor no confunda ambos
documentos por compartir número de artículo.

### Chunking y comparación de embeddings (Fases 2 y 4)

Se probaron 2 configuraciones de chunking y 1 modelo de embeddings local
(`intfloat/multilingual-e5-small`, elegido por estar entrenado para retrieval con
prefijos `query:`/`passage:`, a diferencia de un modelo de similitud genérico).
La comparación completa está en `logs/phase2_chunking_report.md` y
`logs/phase4_evaluation_report.md`.

| Config | tamaño/overlap (chars) | fragmentos | Recall@1 | Recall@3 | Recall@5 |
|---|---|---|---|---|---|
| config_a_articulo_corto | 800/120 | 479 | 0.533 | 0.6 | **0.733** |
| config_b_articulo_largo | 1500/250 | 261 | 0.4-0.6* | 0.6 | 0.6 |

\* medido antes de cambiar de modelo de embeddings; ver `logs/` para el detalle por corrida.

**Configuración seleccionada: `config_a_articulo_corto`** (mejor Recall@5 con evidencia real).

La comparación con `text-embedding-3-small` de OpenAI quedó **pendiente de ejecutar**
porque la cuenta usada no tenía crédito cargado al momento de correr la evaluación
(`insufficient_quota`, ver `logs/phase4_evaluation_report.md`). El código
(`src/embeddings.py::OpenAIEmbedder`, `src/run_phase4_eval.py`) está completo y se
ejecuta automáticamente en cuanto `.env` tenga una key con crédito — basta con correr
`python -m eval.run_eval` de nuevo.

### Umbral de abstención (Fase 3)

Calibrado con evidencia de `eval/eval_set.json`: las preguntas out-of-domain
genuinamente ajenas caen en similitud top-1 ≤ 0.842, mientras que las 15 preguntas
in-domain (incluidas las coloquiales) caen en ≥ 0.852. Umbral operativo:
**0.847** (`config.yaml: rag_engine.similarity_threshold`).

Caso límite documentado: una pregunta "trampa" sobre un procedimiento que la Ley
menciona pero que solo detalla el Reglamento (no indexado) obtiene similitud 0.878,
dentro del rango in-domain — la similitud sola no basta para detectarla. Por eso el
motor tiene una **segunda capa de abstención a nivel de prompt**: el LLM recibe la
instrucción explícita de decir que no encuentra respaldo en los fragmentos en vez de
inventar contenido a partir de material superficialmente relacionado.

### Costos

Cada llamada a OpenAI (chat o embeddings) se registra en `logs/cost_log.jsonl` con
modelo, tokens, costo real (tarifas en `config.yaml: pricing`, fecha de referencia
2026-09-23) y latencia medida. El motor abstiene **antes** de llamar al LLM cuando
la similitud no alcanza el umbral, evitando costo en esos casos.

## Tarea 2 — Cómo correrla

Ver [tarea2_radar/README.md](tarea2_radar/README.md) para el detalle completo
(fuente de datos, arquitectura, resultados de cada fase). Resumen rápido:

```bash
cd tarea2_radar
python build_data.py                # Fases 1-3 (descarga ~200MB, tarda; el índice es resumible)
python -m eval.run_eval             # Fase 4
streamlit run app.py                # Fase 4/5: dashboard
```

## Próximos pasos

- Cargar crédito en la cuenta de OpenAI usada y volver a correr `eval/run_eval.py`
  en ambas tareas para completar la comparación con `text-embedding-3-small` y las
  respuestas reales del motor RAG (por ahora solo se verificó la ruta de abstención,
  que no tiene costo).
- Grabar el video de presentación (máx. 12 min, pipeline antes que código).
