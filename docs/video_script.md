# Guion del video de presentación (máx. 12 min)

Regla que no se puede romper: **explicar el pipeline completo de ambas tareas
antes de mostrar código** (el issue da 0 puntos si el código aparece primero).
Este guion respeta el orden y los tiempos sugeridos. Los números están
sacados directo de los reportes reales del repo — cítalos tal cual, no los
redondees para que suenen mejor.

---

## 0:00 – 1:00 | Problema y contexto de usuario

**Decir:**
> "Este proyecto resuelve dos problemas reales para alguien que necesita
> entender contrataciones públicas en Perú: (1) un funcionario o ciudadano que
> necesita saber qué dice la ley sin leer 63 páginas de la Ley 32069 y sus
> decretos modificatorios, y (2) alguien que quiere auditar en qué se está
> gastando el Estado — por departamento, por categoría — y detectar señales de
> riesgo como adjudicaciones con un solo postor."

**Mostrar en pantalla:** portada con los dos títulos (RAG Normativo / Radar de
Contrataciones) y el logo/nombre del repo.

---

## 1:00 – 3:00 | Pipeline Tarea 1 (diagrama, SIN código)

**Mostrar:** el diagrama Mermaid de `docs/pipeline.md` (Tarea 1). Ábrelo
renderizado en GitHub o en el preview de tu editor — no muestres el .md crudo.

**Decir, siguiendo el diagrama de izquierda a derecha:**
1. "Partimos de dos documentos oficiales: la Ley N.° 32069 descargada de
   gob.pe, y el Decreto Supremo 001-2026-EF descargado de El Peruano. Verificamos
   que ambos son PDF de texto nativo, no escaneados — 0 páginas sospechosas."
2. "Extraemos el texto **por página** con PyMuPDF, porque la página es la
   unidad que después vamos a citar. Limpiamos encabezados y pies de página
   repetidos — muéstrales el antes/después real del reporte de calidad."
3. "Partimos el texto en fragmentos de 800 caracteres con 120 de solape.
   ¿Por qué esa configuración y no una de 1500? Porque la probamos contra la
   otra y dio mejor Recall@5: 0.733 contra 0.6. Esto lo explico con más
   detalle en la sección de decisiones técnicas."
4. "Cada fragmento se embebe con un modelo local — `multilingual-e5-small` —
   y ese índice es lo único que la app carga en producción, nunca se
   reconstruye en caliente."
5. "Cuando alguien pregunta, el motor calcula la similitud ANTES de llamar al
   LLM. Si no llega al umbral calibrado, se abstiene sin gastar nada."

---

## 3:00 – 4:30 | Pipeline Tarea 2 (diagrama, SIN código)

**Mostrar:** diagrama Mermaid de `docs/pipeline.md` (Tarea 2).

**Decir:**
1. "Acá la fuente es el Portal de Contrataciones Abiertas del OECE, que
   publica en el estándar internacional OCDS. Descargamos 3 meses reales de
   2026 (julio, agosto, septiembre) — 17,915 procesos de contratación."
2. "OCDS distingue entre *release* (un evento puntual, como 'se publicó la
   convocatoria') y *record* (la vista consolidada de todas las releases de
   un proceso). Nosotros trabajamos con records para tener una fila por
   proceso; y usamos la API — no la descarga masiva — para traer
   actualizaciones recientes: primero pedimos qué OCID tuvieron releases
   nuevas, y por cada uno pedimos su record actualizado."
3. "El departamento del comprador viene como metadata desde el JSON original
   — no lo inferimos nosotros — pero sí lo normalizamos a los 25
   departamentos oficiales, con reglas documentadas para casos como 'Lima
   Metropolitana' o 'Provincia Constitucional del Callao'."
4. "Acá viene la decisión más importante de esta tarea: el monto y el
   departamento NUNCA se meten al texto que se embebe. Son filtros
   estructurados sobre la tabla. La búsqueda semántica solo indexa la
   descripción del proceso."

---

## 4:30 – 6:30 | Decisiones técnicas con datos de soporte

**Mostrar:** las tablas de `logs/phase2_chunking_report.md` (Tarea 1) y
`logs/phase4_evaluation_report.md` (ambas tareas).

**Explicar con números reales (no inventes ni redondees):**

- **Chunking (Tarea 1):** "800/120 caracteres dio Recall@5=0.733 contra
  0.6 de 1500/250, con el mismo modelo de embeddings. Fragmentos más chicos
  aíslan mejor un artículo específico."
- **Por qué ese modelo de embeddings:** "Usamos `multilingual-e5-small`
  porque está entrenado específicamente para *retrieval* con prefijos
  `query:`/`passage:`, a diferencia de un modelo genérico de similitud. Lo
  probé primero con un modelo genérico y el Recall@1 era 0.4; con este subió
  a 0.6."
- **Umbral de abstención — el punto exacto de la decisión:** "Lo calibro
  comparando la similitud top-1 de preguntas que SÍ están en el corpus contra
  preguntas que NO lo están. En la Tarea 1: preguntas dentro del corpus caen
  en 0.852 o más; preguntas totalmente ajenas, en 0.842 o menos. El umbral
  queda en 0.847. En la Tarea 2 el patrón es el mismo: 0.844."
- **El caso límite que hay que mencionar:** "Hay un tipo de pregunta que ni el
  umbral resuelve bien: una pregunta que SÍ es del dominio pero cuyo detalle
  no está en el corpus indexado (ejemplo: preguntar por un procedimiento que
  la Ley menciona pero que solo detalla el Reglamento, que no descargamos).
  Esa pregunta obtiene una similitud alta (0.878) porque el tema es correcto,
  aunque el detalle no esté. Por eso hay una SEGUNDA capa de abstención: la
  instrucción explícita al LLM de decir 'no encuentro esto' en vez de
  inventar con contenido superficialmente relacionado."
- **Qué significa Recall@k aquí:** "De cada pregunta de prueba con una
  respuesta conocida, Recall@k mide si el fragmento/proceso correcto aparece
  entre los k resultados que devuelve la búsqueda. Recall@5 = 0.733 en Tarea 1
  significa que, de cada 100 preguntas, en 73 el fragmento correcto estaba
  entre los 5 primeros resultados."
- **Costo:** "Cada llamada al LLM se registra con modelo, tokens de entrada y
  salida, y el costo se calcula con las tarifas oficiales de OpenAI vigentes
  al [fecha], guardadas en `config.yaml` para que el cálculo sea auditable si
  cambian los precios. El motor abstiene antes de llamar al LLM cuando no
  hace falta, así que ese costo nunca se paga sin necesidad."

---

## 6:30 – 9:30 | Demostraciones en vivo

**Correr:**
```bash
# Tarea 1
cd tarea1_rag_normativo && streamlit run app.py
# Tarea 2 (en otra terminal)
cd tarea2_radar && streamlit run app.py
```

**Guion de la demo — Tarea 1 (~1.5 min):**
1. Pregunta que SÍ debe responder: *"¿Hasta cuántas UIT se considera un
   contrato menor?"* → muestra la respuesta, la página citada, el costo.
2. Pregunta que debe abstenerse: *"¿Cuál es la tasa del IGV en Perú?"* →
   muestra el banner de abstención y que el costo es $0.
3. Abre la pestaña "Calidad de extracción" y "Evaluación" un segundo para
   mostrar que los reportes están vivos, no son capturas de pantalla.

**Guion de la demo — Tarea 2 (~1.5 min):**
1. Muestra el mapa y los KPIs (17,915 procesos, ~S/ 20 mil millones).
2. Pregunta: *"¿Hay procesos de compra de ambulancias?"* → respuesta citando
   OCID.
3. Filtra por un departamento en el sidebar y muestra cómo cambian el mapa y
   la tabla.
4. Pestaña "Riesgo": muestra el ranking de compradores con mayor % de
   adjudicaciones a un solo postor (dato real: algunas entidades superan 85%).
5. Pregunta trampa: *"¿Hay procesos de compra de submarinos?"* → abstención.

---

## 9:30 – 10:30 | Code walkthrough (2-3 secciones clave, no más)

No muestres todo el repo. Elige:

1. **`src/rag_engine.py` (Tarea 1) o `src/hybrid_rag.py` (Tarea 2):** la
   función `answer_question` — señala en pantalla la línea donde se compara
   `top_similarity < threshold` ANTES del bloque que llama a OpenAI. Es el
   corazón de la abstención.
2. **`src/hybrid_rag.py::apply_filters` (Tarea 2):** muestra que el filtro de
   monto/departamento es una máscara de pandas, no texto que se embebe.
3. **`src/index_store.py` (Tarea 2), la función `build_index`:** señala el
   guardado por lotes con `.progress` — cuéntales brevemente que esto fue
   necesario porque embeber ~18,000 procesos localmente puede tardar 20-30
   minutos, y no querías perder el avance si algo se interrumpía.

---

## 10:30 – 12:00 | Hallazgos, limitaciones y costos reales

**Decir, con honestidad (esto se evalúa):**

- "El Recall@5 real de la Tarea 1 es 0.733, y el de la Tarea 2 es 0.6 — no es
  perfecto. En la Tarea 2, específicamente, con solo 2-6 procesos relevantes
  por pregunta, cada fallo pesa mucho en el porcentaje; aun así el sistema
  encuentra respuestas semánticamente correctas que mi propio set de
  evaluación no había anotado como válidas al principio — tuve que corregir
  mi metodología de evaluación a mitad de camino."
- "Limitación de corpus: la Tarea 1 solo indexa la Ley y un decreto que
  modifica el Reglamento — no el Reglamento completo. Eso genera casos donde
  el sistema reconoce el tema pero no tiene el detalle, y ahí depende de la
  segunda capa de abstención (el LLM), no solo del umbral de similitud."
- "No pude completar la comparación con `text-embedding-3-small` de OpenAI ni
  probar respuestas reales del LLM en producción porque la cuenta de OpenAI
  usada no tenía crédito cargado al momento de correr la evaluación — el
  código está completo y se ejecuta apenas se cargue crédito."
- "Costo real observado: [completa con lo que veas en `logs/cost_log.jsonl`
  de cada tarea una vez que cargues crédito y hagas preguntas reales — antes
  de grabar, corre unas 5-10 preguntas para tener números genuinos que
  mostrar aquí]."
- "Próximo paso natural: enlazar ambas tareas (Tarea 2 podría citar el
  artículo exacto de la Ley 32069 relevante a cada categoría de contratación,
  usando el motor de la Tarea 1) — quedó fuera de alcance por tiempo."

**Cierre:** agradecimiento breve + link al repo en pantalla.

---

## Checklist antes de grabar

- [ ] Cargar crédito en OpenAI y correr `eval/run_eval.py` en ambas tareas
      para tener respuestas reales del LLM que mostrar (opcional pero mejora
      mucho la sección 6:30-9:30 y los costos reales de la sección final).
- [ ] Correr ambas apps una vez ANTES de grabar para que los modelos ya estén
      cacheados (si no, la primera carga del embedder tarda ~1 minuto y se
      ve mal en el video).
- [ ] Tener los diagramas de `docs/pipeline.md` abiertos y renderizados
      (GitHub o tu editor), no el `.md` crudo.
- [ ] Practicar el orden una vez: pipeline → decisiones → demo → código →
      limitaciones. No empieces por el código bajo ningún motivo.
