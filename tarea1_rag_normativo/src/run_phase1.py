"""Orquestador de la Fase 1: para cada documento en config.yaml —
1) verifica la fuente (páginas, chars/página, extraibilidad),
2) extrae texto por página,
3) limpia el texto con reglas documentadas,
4) escribe página limpia + cruda a data/processed/*.jsonl,
5) genera el reporte de calidad por documento y un resumen consolidado.

Idempotente: se puede re-ejecutar sin efectos secundarios (sobrescribe salidas).
Uso: python -m src.run_phase1   (ejecutado desde tarea1_rag_normativo/)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.cleaning import clean_page_text, make_before_after_examples
from src.config import load_config, resolve_path
from src.pdf_extraction import extract_pages, verify_source


def run_phase1() -> dict:
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    logs_dir = resolve_path(cfg["paths"]["logs_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    min_chars_ok = cfg["extraction"]["min_chars_per_page_ok"]
    boilerplate = cfg["cleaning"]["boilerplate_patterns"]
    join_hyphen = cfg["cleaning"]["join_hyphenated_linebreaks"]
    collapse_ws = cfg["cleaning"]["collapse_whitespace"]
    strip_ctrl = cfg["cleaning"]["strip_control_chars"]

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "documents": [],
    }

    for doc_cfg in cfg["documents"]:
        doc_id = doc_cfg["id"]
        pdf_path = raw_dir / doc_cfg["file"]
        if not pdf_path.exists():
            raise FileNotFoundError(f"No se encontró el PDF fuente: {pdf_path}")

        # 1) Verificación de fuente
        verification = verify_source(pdf_path, doc_id, min_chars_ok)

        # 2) Extracción por página
        pages = extract_pages(pdf_path)

        # 3) Limpieza + escritura de fragmentos crudos y limpios por página
        raw_out_path = processed_dir / f"{doc_id}_pages_raw.jsonl"
        clean_out_path = processed_dir / f"{doc_id}_pages_clean.jsonl"

        cleaned_pages = []
        with open(raw_out_path, "w", encoding="utf-8") as f_raw, \
             open(clean_out_path, "w", encoding="utf-8") as f_clean:
            for p in pages:
                cleaned_text = clean_page_text(
                    p.text,
                    boilerplate_patterns=boilerplate,
                    join_hyphenated_linebreaks=join_hyphen,
                    collapse_whitespace=collapse_ws,
                    strip_control_chars=strip_ctrl,
                )
                cleaned_pages.append((p.page_number, cleaned_text))

                f_raw.write(json.dumps({
                    "doc_id": doc_id,
                    "page_number": p.page_number,
                    "char_count": p.char_count,
                    "text": p.text,
                }, ensure_ascii=False) + "\n")

                f_clean.write(json.dumps({
                    "doc_id": doc_id,
                    "page_number": p.page_number,
                    "char_count": len(cleaned_text),
                    "text": cleaned_text,
                }, ensure_ascii=False) + "\n")

        # 4) Ejemplos de antes/después para el reporte
        raw_pairs = [(p.page_number, p.text) for p in pages]
        examples = make_before_after_examples(raw_pairs, boilerplate, max_examples=3)

        # 5) Reporte de calidad por documento (Markdown, legible en el README/Streamlit)
        report_path = logs_dir / f"quality_report_{doc_id}.md"
        _write_quality_report(
            report_path=report_path,
            doc_cfg=doc_cfg,
            verification=verification,
            examples=examples,
        )

        summary["documents"].append({
            "doc_id": doc_id,
            "file": doc_cfg["file"],
            "source_url": doc_cfg["source_url"],
            "source_portal": doc_cfg["source_portal"],
            "page_count": verification.page_count,
            "is_text_extractable": verification.is_text_extractable,
            "suspicious_pages": verification.suspicious_pages,
            "min_chars_per_page": verification.min_chars,
            "max_chars_per_page": verification.max_chars,
            "mean_chars_per_page": round(verification.mean_chars, 1),
            "median_chars_per_page": verification.median_chars,
            "embedded_image_count": verification.embedded_image_count,
            "quality_report": str(report_path.relative_to(resolve_path("."))),
            "raw_output": str(raw_out_path.relative_to(resolve_path("."))),
            "clean_output": str(clean_out_path.relative_to(resolve_path("."))),
        })

    summary_path = logs_dir / "phase1_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def _write_quality_report(report_path: Path, doc_cfg: dict, verification, examples: list[dict]) -> None:
    lines = []
    lines.append(f"# Reporte de calidad — {doc_cfg['title']}\n")
    lines.append(f"- **Documento (id):** `{doc_cfg['id']}`")
    lines.append(f"- **Archivo:** `{doc_cfg['file']}`")
    lines.append(f"- **Fuente:** {doc_cfg['source_portal']} — {doc_cfg['source_url']}")
    lines.append(f"- **Versión:** {doc_cfg['version_label']} ({doc_cfg['version_date']})")
    lines.append(f"- **Rol del documento:** {doc_cfg['document_role']}")
    lines.append("")
    lines.append("## Verificación de fuente")
    lines.append(f"- Páginas totales: **{verification.page_count}**")
    lines.append(f"- Caracteres por página — mín: {verification.min_chars}, "
                  f"máx: {verification.max_chars}, "
                  f"media: {verification.mean_chars:.1f}, "
                  f"mediana: {verification.median_chars}")
    lines.append(f"- Imágenes embebidas detectadas: {verification.embedded_image_count} "
                  "(0 esperado si el PDF es texto nativo, no un escaneo)")
    extractable_txt = "SÍ (texto nativo, extracción confiable)" if verification.is_text_extractable else "NO (revisar / posible escaneo, requeriría OCR)"
    lines.append(f"- ¿Texto extraíble de forma confiable?: **{extractable_txt}**")
    if verification.suspicious_pages:
        lines.append(f"- ⚠️ Páginas con menos de umbral de caracteres (posible página en blanco, "
                      f"separata o solo imagen): {verification.suspicious_pages}")
    else:
        lines.append("- Sin páginas sospechosas por debajo del umbral mínimo de caracteres.")
    lines.append("")
    lines.append("## Ejemplos de limpieza (antes → después)")
    if not examples:
        lines.append("_No se detectaron cambios de limpieza en las páginas muestreadas._")
    for i, ex in enumerate(examples, start=1):
        lines.append(f"\n**Ejemplo {i} — página {ex['page_number']}**\n")
        lines.append("Antes:")
        lines.append("```")
        lines.append(ex["before"])
        lines.append("```")
        lines.append("Después:")
        lines.append("```")
        lines.append(ex["after"])
        lines.append("```")

    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = run_phase1()
    print(json.dumps(result, ensure_ascii=False, indent=2))
