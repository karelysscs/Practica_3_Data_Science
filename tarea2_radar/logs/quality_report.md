# Reporte de calidad — Tarea 2 (Radar de Contrataciones)

## Adquisición (Fase 1)
- Registros antes de deduplicar por OCID: **17915**
- Registros tras concatenar fuentes: **17915**
- Registros tras deduplicar por OCID (1 fila = 1 proceso): **17915**
- Duplicados de OCID removidos: **0**

## Validación (Fase 2)
- Total de procesos: **17915**
- Monto de la convocatoria ausente o en cero: **4790** (26.74%) — regla: `tender_value_amount es nulo o <= 0.0`
- Descripción ausente: **0** (0.0%)
- Departamento del comprador ausente en la fuente: **0**
- Valores que NO son uno de los 25 departamentos tras normalizar: **0**
- Regla de normalización: Se pasa a mayúsculas, se quitan tildes/espacios extra y se compara contra los 25 departamentos oficiales; alias conocidos (LIMA METROPOLITANA, PROVINCIA CONSTITUCIONAL DEL CALLAO, etc., ver src/departments.py) se mapean al departamento canónico. Lo que no matchea queda marcado inválido.
- Registros con texto reparado por problema de codificación: **0**
- Registros con codificación aún sospechosa tras el intento de reparación: **2**
- Regla: Se detecta texto con patrón de doble-codificación (UTF-8 leído como Latin-1, ej. 'MUÃ¿OZ') y se repara con encode('latin1').decode('utf-8') cuando el resultado ya no contiene el patrón.