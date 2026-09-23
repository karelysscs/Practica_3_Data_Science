"""Fase 1 — Reglas de limpieza de texto extraído.

Cada regla está documentada aquí mismo (nombre, motivo, patrón) para poder
generar ejemplos de antes/después de forma reproducible en el reporte de calidad.
Este módulo tampoco importa UI.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass
class CleaningRule:
    name: str
    reason: str
    pattern: str  # regex, aplicada línea por línea salvo que se indique lo contrario


def build_rules(boilerplate_patterns: list[str]) -> list[CleaningRule]:
    rules = [
        CleaningRule(
            name="remove_boilerplate_line",
            reason=(
                "El Diario Oficial y las colecciones de gob.pe repiten encabezados/pies de "
                "página (título de sección, nombre del diario, fecha, número de página aislado) "
                "en cada página. Esto no es contenido normativo y contamina los fragmentos."
            ),
            pattern="|".join(f"(?:{p})" for p in boilerplate_patterns),
        ),
        CleaningRule(
            name="strip_control_chars",
            reason="Caracteres de control (\\r, \\x0c salto de página, etc.) quedan de la extracción del PDF.",
            pattern=r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
        ),
    ]
    return rules


_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")


def clean_page_text(
    raw_text: str,
    boilerplate_patterns: list[str],
    join_hyphenated_linebreaks: bool = True,
    collapse_whitespace: bool = True,
    strip_control_chars: bool = True,
) -> str:
    """Aplica, en orden, las reglas de limpieza documentadas en build_rules()."""
    text = unicodedata.normalize("NFC", raw_text)

    if strip_control_chars:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)

    if join_hyphenated_linebreaks:
        # "adelan-\ntos" -> "adelantos" (corte de palabra por salto de línea del PDF)
        text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    if boilerplate_patterns:
        combined = re.compile("|".join(f"(?:{p})" for p in boilerplate_patterns), re.IGNORECASE)
        lines = text.split("\n")
        lines = [ln for ln in lines if not combined.fullmatch(ln.strip())]
        text = "\n".join(lines)

    if collapse_whitespace:
        text = _MULTI_SPACE_RE.sub(" ", text)
        text = _MULTI_BLANK_LINE_RE.sub("\n\n", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = text.strip()

    return text


def make_before_after_examples(
    raw_pages: list[tuple[int, str]],
    boilerplate_patterns: list[str],
    max_examples: int = 3,
) -> list[dict]:
    """Genera ejemplos concretos de antes/después para el reporte de calidad,
    eligiendo páginas donde la limpieza realmente cambió algo."""
    examples = []
    for page_number, raw in raw_pages:
        cleaned = clean_page_text(raw, boilerplate_patterns)
        if cleaned != raw.strip() and len(examples) < max_examples:
            examples.append(
                {
                    "page_number": page_number,
                    "before": raw[:400],
                    "after": cleaned[:400],
                }
            )
    return examples
