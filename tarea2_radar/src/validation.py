"""Fase 2 — Validación y normalización de la tabla de procesos.

Detecta y registra (regla explícita del issue):
- Registros repetidos para el mismo proceso (ya resuelto en ocds_parse, se
  reporta aquí el conteo antes/después).
- Montos ausentes o en cero.
- Descripciones ausentes.
- Valores de ubicación que no son un departamento válido.
- Inconsistencias de codificación de texto (mojibake reparado).

Normaliza la ubicación del comprador a los 25 departamentos (src/departments.py)
con reglas documentadas. Produce un reporte de calidad (reglas aplicadas +
conteos). No importa UI.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.departments import CANONICAL_DEPARTMENTS, normalize_department

_RESIDUAL_MOJIBAKE_RE = re.compile(r"[ÃÂ]")


def validate_and_normalize(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    # --- Normalización de departamento -----------------------------------
    normalized = df["buyer_department_raw"].apply(normalize_department)
    df["buyer_department"] = normalized.apply(lambda t: t[0])
    df["department_is_valid"] = normalized.apply(lambda t: t[1])

    # --- Métricas de calidad -----------------------------------------------
    threshold = cfg["validation"]["zero_amount_threshold"]
    n = len(df)

    missing_or_zero_amount = int(((df["tender_value_amount"].isna()) | (df["tender_value_amount"] <= threshold)).sum())
    missing_description = int((df["tender_description"].isna() | (df["tender_description"].str.strip() == "")).sum())
    invalid_department = int((~df["department_is_valid"]).sum())
    missing_department = int(df["buyer_department_raw"].isna().sum())
    encoding_fixed = int(df["encoding_fixed"].sum())

    residual_mojibake = 0
    for col in ("tender_title", "tender_description", "buyer_name"):
        residual_mojibake += int(df[col].fillna("").apply(lambda s: bool(_RESIDUAL_MOJIBAKE_RE.search(s))).sum())

    invalid_dept_examples = (
        df.loc[~df["department_is_valid"], "buyer_department_raw"].dropna().unique().tolist()[:10]
    )

    build_stats = df.attrs.get("build_stats", {})

    report = {
        "total_processes": n,
        "acquisition": build_stats,
        "missing_or_zero_tender_amount": {
            "count": missing_or_zero_amount,
            "pct": round(100 * missing_or_zero_amount / n, 2) if n else 0,
            "rule": f"tender_value_amount es nulo o <= {threshold}",
        },
        "missing_description": {
            "count": missing_description,
            "pct": round(100 * missing_description / n, 2) if n else 0,
        },
        "location_department": {
            "missing_raw_value": missing_department,
            "invalid_after_normalization": invalid_department,
            "invalid_examples": invalid_dept_examples,
            "canonical_department_count": len(CANONICAL_DEPARTMENTS),
            "rule": (
                "Se pasa a mayúsculas, se quitan tildes/espacios extra y se compara contra "
                "los 25 departamentos oficiales; alias conocidos (LIMA METROPOLITANA, "
                "PROVINCIA CONSTITUCIONAL DEL CALLAO, etc., ver src/departments.py) se "
                "mapean al departamento canónico. Lo que no matchea queda marcado inválido."
            ),
        },
        "text_encoding": {
            "records_with_mojibake_fixed": encoding_fixed,
            "residual_mojibake_after_fix": residual_mojibake,
            "rule": (
                "Se detecta texto con patrón de doble-codificación (UTF-8 leído como "
                "Latin-1, ej. 'MUÃ¿OZ') y se repara con encode('latin1').decode('utf-8') "
                "cuando el resultado ya no contiene el patrón."
            ),
        },
    }
    return df, report


def write_quality_report_markdown(report: dict, out_path: Path) -> None:
    lines = ["# Reporte de calidad — Tarea 2 (Radar de Contrataciones)\n"]
    acq = report["acquisition"]
    lines.append("## Adquisición (Fase 1)")
    lines.append(f"- Registros antes de deduplicar por OCID: **{acq.get('rows_before_dedup')}**")
    lines.append(f"- Registros tras concatenar fuentes: **{acq.get('rows_after_raw_concat')}**")
    lines.append(f"- Registros tras deduplicar por OCID (1 fila = 1 proceso): **{acq.get('rows_after_dedup_by_ocid')}**")
    lines.append(f"- Duplicados de OCID removidos: **{acq.get('duplicated_ocids_removed')}**")

    lines.append("\n## Validación (Fase 2)")
    lines.append(f"- Total de procesos: **{report['total_processes']}**")

    ma = report["missing_or_zero_tender_amount"]
    lines.append(f"- Monto de la convocatoria ausente o en cero: **{ma['count']}** ({ma['pct']}%) — regla: `{ma['rule']}`")

    md = report["missing_description"]
    lines.append(f"- Descripción ausente: **{md['count']}** ({md['pct']}%)")

    loc = report["location_department"]
    lines.append(f"- Departamento del comprador ausente en la fuente: **{loc['missing_raw_value']}**")
    lines.append(f"- Valores que NO son uno de los 25 departamentos tras normalizar: **{loc['invalid_after_normalization']}**")
    if loc["invalid_examples"]:
        lines.append(f"  - Ejemplos: {loc['invalid_examples']}")
    lines.append(f"- Regla de normalización: {loc['rule']}")

    enc = report["text_encoding"]
    lines.append(f"- Registros con texto reparado por problema de codificación: **{enc['records_with_mojibake_fixed']}**")
    lines.append(f"- Registros con codificación aún sospechosa tras el intento de reparación: **{enc['residual_mojibake_after_fix']}**")
    lines.append(f"- Regla: {enc['rule']}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
