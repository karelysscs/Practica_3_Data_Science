# Fase 4 — Evaluación y comparación de embeddings

Configuración de chunking evaluada: `config_a_articulo_corto`

## Comparación local vs. OpenAI

| Embedder | Dim. | Fragmentos | Recall@1 | Recall@3 | Recall@5 | Tiempo indexado (s) | Costo indexado (USD) | Latencia media consulta (s) |
|---|---|---|---|---|---|---|---|---|
| local | 384 | 479 | 0.533 | 0.6 | 0.733 | 0.0 | 0.0 | 0.0384 |
| openai | — | — | — | — | — | — | — | _omitido: Falló la llamada a la API de OpenAI: RateLimitError: Error code: 429 - {'error': {'message': 'You have no credits remaining. Add credits to continue using the API at https://platform.openai.com/settings/organization/billing/.', 'type': 'insufficient_quota', 'param': None, 'code': 'credit_balance_exhausted'}}_ |

## Calibración del umbral de abstención (con el índice local en producción)

- Similitud top-1 mínima entre preguntas in-domain: **0.8521**
- Similitud top-1 máxima entre preguntas out-of-domain (todas): **0.878** (pregunta `q20`, caso límite deliberado)
- Similitud top-1 máxima entre preguntas out-of-domain genuinamente ajenas: **0.8416**
- Umbral operativo elegido (evidencia): **0.847**
- Umbral configurado actualmente en config.yaml: **0.845**

La pregunta out-of-domain con mayor similitud es **q20** (sim=0.878): Trampa de dominio: la Ley (Art. 56.1) menciona que ese procedimiento 'se rige por lo señalado en el reglamento', pero el Reglamento en sí no está en el corpus indexado (solo el DS que modifica algunos de sus artículos, ninguno de ellos éste). El motor debe reconocer que no tiene el detalle y abstenerse, no inventar el procedimiento. Es un caso límite deliberado (tema del dominio pero detalle fuera del corpus indexado), no una pregunta genuinamente ajena; por eso se calibra el umbral con el resto de preguntas out-of-domain (máx.=0.8416) en vez de con este outlier. Umbral operativo elegido: 0.847 (vs. 0.8521 mínimo in-domain). Con este umbral, **q20 sigue sin abstenerse por similitud** — queda como responsabilidad de la segunda capa de abstención (instrucción explícita al LLM de no responder sin respaldo en los fragmentos), documentado como limitación conocida del sistema.
