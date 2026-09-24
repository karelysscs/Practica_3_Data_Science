"""Fase 1/2 — De records OCDS (release vs. record) a una tabla plana con
UNA FILA POR PROCESO DE CONTRATACIÓN (identificado por OCID).

Terminología OCDS (documentada aquí porque el issue pide explícitamente
entenderla, no solo usarla):
- "release": un evento/publicación puntual de un proceso (p.ej. "se publicó la
  convocatoria", "se otorgó la buena pro"). Un mismo proceso genera varias
  releases a lo largo del tiempo.
- "record": la vista CONSOLIDADA y vigente de todas las releases de un OCID
  (aquí, `compiledRelease`), más la lista de las releases que la componen
  (`releases`, con su url/fecha/tag). Es lo que usamos para construir una fila
  por proceso: tomar el compiledRelease de cada record.

No importa UI.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


def fix_mojibake(text: str | None) -> tuple[str | None, bool]:
    """Intenta reparar texto que fue decodificado con la codificación
    equivocada (UTF-8 leído como Latin-1), un problema real observado en el
    corpus (ej. 'MUÃ¿OZ' en vez de 'MUÑOZ'). Devuelve (texto, se_corrigio)."""
    if not text:
        return text, False
    if not re.search(r"[ÃÂ]", text):
        return text, False
    try:
        fixed = text.encode("latin1").decode("utf-8")
        if fixed != text and not re.search(r"[ÃÂ]", fixed):
            return fixed, True
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return text, False


def _buyer_info(compiled: dict) -> dict:
    for p in compiled.get("parties", []):
        roles = p.get("roles") or []
        if "buyer" in roles or "procuringEntity" in roles:
            addr = p.get("address") or {}
            return {
                "buyer_id": p.get("id"),
                "buyer_name": p.get("name"),
                "buyer_department_raw": addr.get("department"),
            }
    return {"buyer_id": None, "buyer_name": None, "buyer_department_raw": None}


def _tenderer_stats(compiled: dict) -> dict:
    tenderers = [p for p in compiled.get("parties", []) if "tenderer" in (p.get("roles") or [])]
    return {"num_tenderers": len(tenderers)}


def _award_stats(compiled: dict) -> dict:
    awards = compiled.get("awards") or []
    total_award_value = 0.0
    currency = None
    suppliers = []
    for a in awards:
        val = a.get("value") or {}
        total_award_value += val.get("amount") or 0.0
        currency = currency or val.get("currency")
        for s in a.get("suppliers", []):
            suppliers.append(s.get("name"))
    return {
        "num_awards": len(awards),
        "award_value_amount": total_award_value if awards else None,
        "award_currency": currency,
        "winning_suppliers": "; ".join(dict.fromkeys(suppliers)) if suppliers else None,
    }


def record_to_row(record: dict, source_label: str) -> dict:
    """Convierte UN record OCDS (compiledRelease + releases) en una fila plana."""
    compiled = record["compiledRelease"]
    tender = compiled.get("tender", {}) or {}
    value = tender.get("value", {}) or {}

    buyer = _buyer_info(compiled)
    tenderer_stats = _tenderer_stats(compiled)
    award_stats = _award_stats(compiled)

    title, title_fixed = fix_mojibake(tender.get("title"))
    description, desc_fixed = fix_mojibake(tender.get("description"))
    buyer_name, buyer_fixed = fix_mojibake(buyer["buyer_name"])

    row = {
        "ocid": record["ocid"],
        "source_month": source_label,
        "n_releases_in_record": len(record.get("releases", [])),  # evidencia release vs record
        "buyer_id": buyer["buyer_id"],
        "buyer_name": buyer_name,
        "buyer_department_raw": buyer["buyer_department_raw"],
        "tender_title": title,
        "tender_description": description,
        "procurement_method": tender.get("procurementMethod"),
        "procurement_method_details": tender.get("procurementMethodDetails"),
        "main_category": tender.get("mainProcurementCategory"),
        "tender_value_amount": value.get("amount"),
        "tender_currency": value.get("currency"),
        "tender_date_published": tender.get("datePublished"),
        "num_tenderers": tenderer_stats["num_tenderers"],
        "is_single_bidder_award": (
            tenderer_stats["num_tenderers"] == 1 if award_stats["num_awards"] > 0 else None
        ),
        **award_stats,
        "encoding_fixed": bool(title_fixed or desc_fixed or buyer_fixed),
    }
    return row


def load_records_from_bulk_file(json_path: Path) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["records"]


def build_processes_table(bulk_file_paths: list[Path], extra_records: list[dict] | None = None) -> pd.DataFrame:
    """Construye la tabla de UNA FILA POR PROCESO a partir de los archivos bulk
    mensuales (+ opcionalmente records recientes traídos por la API). Si un OCID
    aparece más de una vez (por ejemplo, porque la API trajo una actualización
    más reciente de un proceso ya presente en el bulk), se conserva la versión
    con más releases en su historial (la más completa/actualizada) — así queda
    resuelta la deduplicación por proceso de forma determinista."""
    rows_before = 0
    rows = []
    for path in bulk_file_paths:
        month_label = path.stem.split("_")[0]  # "2026-07"
        records = load_records_from_bulk_file(path)
        rows_before += len(records)
        for rec in records:
            rows.append(record_to_row(rec, source_label=month_label))

    if extra_records:
        rows_before += len(extra_records)
        for rec in extra_records:
            rows.append(record_to_row(rec, source_label="api_recent"))

    df = pd.DataFrame(rows)
    rows_after_raw = len(df)

    # Deduplicación por OCID: conserva la fila con más releases en su historial
    # (la vista más completa), documentando el conteo antes/después (Fase 1).
    df = df.sort_values("n_releases_in_record", ascending=False).drop_duplicates(subset=["ocid"], keep="first")
    df = df.sort_values("ocid").reset_index(drop=True)

    build_stats = {
        "rows_before_dedup": rows_before,
        "rows_after_raw_concat": rows_after_raw,
        "rows_after_dedup_by_ocid": len(df),
        "duplicated_ocids_removed": rows_after_raw - len(df),
    }
    df.attrs["build_stats"] = build_stats
    return df
