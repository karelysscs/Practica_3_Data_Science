# Fase 2 — Comparación de configuraciones de chunking

| Config | tamaño | overlap | fragmentos | IDs únicos | min | mediana | media | max | stdev |
|---|---|---|---|---|---|---|---|---|---|
| config_a_articulo_corto | 800 | 120 | 479 | ✅ | 219 | 800 | 797.7 | 800 | 31.0 |
| config_b_articulo_largo | 1500 | 250 | 261 | ✅ | 327 | 1500 | 1493.1 | 1500 | 80.1 |

**Configuración seleccionada para Fases 3-5:** `config_a_articulo_corto`


Criterio de selección: ambas configuraciones se indexaron y se corrieron contra las 20 preguntas de `eval/eval_set.json` (Fase 4). Con el modelo de embeddings local (`intfloat/multilingual-e5-small`), fragmentos más CORTOS (config_a, 800/120) dieron mejor Recall@5 (0.733) que fragmentos más largos (config_b, 1500/250: Recall@5=0.6): fragmentos más pequeños aíslan mejor un artículo/numeral específico, lo que ayuda cuando la pregunta (sobre todo las coloquiales) apunta a un dato puntual dentro de un artículo largo. Ver el detalle completo, incluida la calibración del umbral de abstención, en `logs/phase4_evaluation_report.md` y `eval/results_embeddings_comparison.json`.