"""Fase 1 — Verificación de fuentes y extracción de texto por página.

Reglas de arquitectura: este módulo no importa Streamlit ni ninguna librería
de UI. Solo lectura de PDF (PyMuPDF) y estructuras de datos puras.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf  # fitz


@dataclass
class PageExtraction:
    page_number: int  # 1-indexed, tal como aparece impreso/citable
    char_count: int
    text: str


@dataclass
class SourceVerification:
    doc_id: str
    file_name: str
    page_count: int
    chars_per_page: list[int]
    suspicious_pages: list[int]  # páginas por debajo del umbral mínimo de caracteres
    min_chars: int
    max_chars: int
    mean_chars: float
    median_chars: float
    is_text_extractable: bool  # heurística: mediana de chars/página razonable
    embedded_image_count: int


def verify_source(pdf_path: Path, doc_id: str, min_chars_per_page_ok: int) -> SourceVerification:
    """Revisa página por página: cuántos caracteres extrae PyMuPDF y si hay
    señales de que el PDF es en realidad un escaneo (texto vacío / puras imágenes)."""
    doc = pymupdf.open(pdf_path)
    chars_per_page: list[int] = []
    suspicious: list[int] = []
    image_count = 0

    for i in range(doc.page_count):
        page = doc[i]
        text = page.get_text()
        n = len(text.strip())
        chars_per_page.append(n)
        image_count += len(page.get_images())
        if n < min_chars_per_page_ok:
            suspicious.append(i + 1)  # 1-indexed

    doc.close()

    median_chars = statistics.median(chars_per_page) if chars_per_page else 0
    is_extractable = median_chars >= min_chars_per_page_ok

    return SourceVerification(
        doc_id=doc_id,
        file_name=pdf_path.name,
        page_count=len(chars_per_page),
        chars_per_page=chars_per_page,
        suspicious_pages=suspicious,
        min_chars=min(chars_per_page) if chars_per_page else 0,
        max_chars=max(chars_per_page) if chars_per_page else 0,
        mean_chars=statistics.mean(chars_per_page) if chars_per_page else 0.0,
        median_chars=median_chars,
        is_text_extractable=is_extractable,
        embedded_image_count=image_count,
    )


def extract_pages(pdf_path: Path) -> list[PageExtraction]:
    """Extrae el texto de cada página, preservando el número de página (1-indexed)
    tal como debe citarse ante el usuario final."""
    doc = pymupdf.open(pdf_path)
    pages = []
    for i in range(doc.page_count):
        text = doc[i].get_text()
        pages.append(PageExtraction(page_number=i + 1, char_count=len(text), text=text))
    doc.close()
    return pages
