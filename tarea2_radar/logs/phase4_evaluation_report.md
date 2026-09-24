# Fase 4 — Evaluación del RAG híbrido (Tarea 2)

Índice: 17915 procesos, dimensión 384

| k | Recall@k |
|---|---|
| 1 | 0.4 |
| 3 | 0.6 |
| 5 | 0.6 |
| 8 | 0.6 |

Latencia media de consulta: 0.0397s

## Calibración del umbral de abstención
- Similitud top-1 mínima in-scope: **0.8317**
- Similitud top-1 máxima out-of-scope: **0.857**
- Separación limpia: **False**
- Umbral sugerido: **0.844**
- Umbral configurado en config.yaml: **0.844**