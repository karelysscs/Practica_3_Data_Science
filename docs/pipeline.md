# Diagramas de pipeline

Diagramas en Mermaid (se renderizan directo en GitHub). Pensados para
mostrarse en pantalla durante el video **antes** de mostrar código, tal como
exige el issue ("pipeline-first rule").

## Tarea 1 — RAG Normativo

```mermaid
flowchart TD
    subgraph OFFLINE["Proceso OFFLINE — build_index.py"]
        A1["Fuentes oficiales<br/>gob.pe + El Peruano"] -->|descarga| A2["PDF crudos<br/>data/raw/"]
        A2 -->|"Fase 1: PyMuPDF<br/>por página"| A3["Texto extraído<br/>+ verificación de fuente"]
        A3 -->|"limpieza documentada<br/>(encabezados, guiones, control chars)"| A4["Texto limpio<br/>por página"]
        A4 -->|"Fase 2: chunking<br/>800/120 chars (elegido con evidencia)"| A5["Fragmentos<br/>+ metadata: doc, versión, página, ID estable"]
        A5 -->|"embeddings locales<br/>intfloat/multilingual-e5-small"| A6["Índice vectorial<br/>vectors.npy + fragments.jsonl"]
    end

    subgraph ONLINE["Proceso ONLINE — app.py (Streamlit)"]
        B1["Pregunta del usuario"] --> B2["Embedding de la pregunta<br/>(prefijo query:)"]
        B2 --> B3{"Similitud top-1<br/>≥ 0.847?"}
        B3 -- "No" --> B4["🛑 Abstiene<br/>costo = $0, sin llamar al LLM"]
        B3 -- "Sí" --> B5["Construye contexto<br/>con los fragmentos + cita doc/página"]
        B5 --> B6["LLM (gpt-4o-mini)<br/>2ª capa de abstención vía prompt"]
        B6 --> B7["Respuesta + citas<br/>+ costo real registrado"]
    end

    A6 -.->|"índice ya construido,<br/>NO se reconstruye"| B2

    subgraph EVAL["Fase 4 — Evaluación"]
        C1["20 preguntas<br/>15 in-domain + 5 out-of-domain"] --> C2["Recall@k por config<br/>de chunking y embeddings"]
        C2 --> C3["Calibración del umbral<br/>con evidencia real"]
    end
```

**Puntos clave para el video:**
- La página se preserva desde la extracción (Fase 1) hasta la cita final — es
  la unidad de trazabilidad de todo el sistema normativo.
- El umbral de abstención (0.847) se calibró comparando la similitud de
  preguntas dentro y fuera del corpus — no es un número arbitrario.
- Hay DOS capas de abstención: una barata (similitud, antes del LLM) y una
  semántica (instrucción al LLM), porque la primera no basta para el caso
  "pregunta del dominio pero fuera de lo indexado" (ver `eval_set.json`, q20).

## Tarea 2 — Radar de Contrataciones

```mermaid
flowchart TD
    subgraph OFFLINE["Proceso OFFLINE — build_data.py"]
        D1["Portal OECE<br/>contratacionesabiertas.oece.gob.pe"] -->|"descarga masiva<br/>3 meses de 2026"| D2["JSON OCDS crudo<br/>(RECORD packages)"]
        D1 -->|"API: /releases?dateFrom=<br/>(actualizaciones recientes)"| D2b["OCID recientes<br/>-> /record/{ocid}"]
        D2 --> D3["1 fila = 1 proceso<br/>(dedup por OCID)"]
        D2b --> D3
        D3 -->|"Fase 2: validación"| D4["Normalización a<br/>25 departamentos + reporte de calidad"]
        D4 -->|"embeddings SOLO de<br/>título+descripción"| D5["Índice vectorial<br/>(resumible por lotes)"]
    end

    subgraph ONLINE["Dashboard — app.py (Streamlit)"]
        E1["Pregunta + filtros<br/>(departamento, monto, fecha)"] --> E2["Filtros ESTRUCTURADOS<br/>sobre la tabla (pandas)"]
        E2 --> E3["Búsqueda semántica<br/>SOLO sobre el subconjunto filtrado"]
        E3 --> E4{"Similitud ≥ 0.844?"}
        E4 -- "No" --> E5["🛑 Abstiene"]
        E4 -- "Sí" --> E6["LLM + cita por OCID"]
        D4 --> E7["Mapa, KPIs,<br/>tabla, gráficos, riesgo"]
    end

    D5 -.-> E3

    subgraph RISK["Fase 5 — Riesgo"]
        F1["Adjudicaciones<br/>(awards)"] --> F2["% con un solo postor<br/>por departamento y comprador"]
    end
```

**Puntos clave para el video:**
- El departamento del comprador es metadata desde la Fase 1 (viene en el JSON
  OCDS: `party.address.department`) — se normaliza en la Fase 2, no se infiere.
- Monto y departamento **nunca** se meten al texto que se embebe: son filtros
  de pandas. Si se metieran al embedding, "más de S/ 200,000" no se podría
  resolver de forma exacta (la similitud semántica no entiende desigualdades).
- El indicador de riesgo (single-bidder share) usa las mismas adjudicaciones
  reales del corpus — no es una simulación.
