"""Fase 2 — Chunking con metadata de documento, versión y página.

Estrategia: se concatena el texto limpio de todas las páginas de un documento
(en orden), llevando un mapa de qué rango de caracteres pertenece a qué página.
Sobre ese texto continuo se aplica una ventana deslizante (chunk_size, overlap).
Esto evita cortar artículos exactamente en el salto de página y permite que un
fragmento cite un rango de páginas cuando cruza el límite.

No importa UI.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Fragment:
    fragment_id: str
    doc_id: str
    doc_title: str
    version_label: str
    version_date: str
    document_role: str
    page_start: int
    page_end: int
    char_count: int
    chunk_config: str
    text: str


def _load_clean_pages(clean_jsonl_path: Path) -> list[dict]:
    pages = []
    with open(clean_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pages.append(json.loads(line))
    pages.sort(key=lambda p: p["page_number"])
    return pages


def _build_document_stream(pages: list[dict]) -> tuple[str, list[tuple[int, int, int]]]:
    """Devuelve (texto_concatenado, boundaries) donde boundaries es una lista de
    (start_offset, end_offset, page_number) en el texto concatenado."""
    parts = []
    boundaries = []
    offset = 0
    for p in pages:
        text = p["text"]
        start = offset
        parts.append(text)
        offset += len(text)
        # separador entre páginas para no pegar la última palabra de una página
        # con la primera de la siguiente
        parts.append("\n")
        offset += 1
        end = offset
        boundaries.append((start, end, p["page_number"]))
    return "".join(parts), boundaries


def _page_range_for_offset(boundaries: list[tuple[int, int, int]], start: int, end: int) -> tuple[int, int]:
    pages_touched = [pg for (s, e, pg) in boundaries if e > start and s < end]
    if not pages_touched:
        # fallback: última página conocida
        return boundaries[-1][2], boundaries[-1][2]
    return min(pages_touched), max(pages_touched)


def chunk_document(
    doc_cfg: dict,
    clean_jsonl_path: Path,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    chunk_config_name: str,
) -> list[Fragment]:
    pages = _load_clean_pages(clean_jsonl_path)
    full_text, boundaries = _build_document_stream(pages)

    fragments: list[Fragment] = []
    step = max(1, chunk_size_chars - chunk_overlap_chars)
    idx = 0
    seq = 0
    n = len(full_text)
    while idx < n:
        end = min(idx + chunk_size_chars, n)
        chunk_text = full_text[idx:end].strip()
        if chunk_text:
            page_start, page_end = _page_range_for_offset(boundaries, idx, end)
            content_hash = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()[:8]
            fragment_id = f"{doc_cfg['id']}::p{page_start:03d}-{page_end:03d}::{seq:04d}::{content_hash}"
            fragments.append(Fragment(
                fragment_id=fragment_id,
                doc_id=doc_cfg["id"],
                doc_title=doc_cfg["title"],
                version_label=doc_cfg["version_label"],
                version_date=doc_cfg["version_date"],
                document_role=doc_cfg["document_role"],
                page_start=page_start,
                page_end=page_end,
                char_count=len(chunk_text),
                chunk_config=chunk_config_name,
                text=chunk_text,
            ))
            seq += 1
        if end == n:
            break
        idx += step

    return fragments


def write_fragments_jsonl(fragments: list[Fragment], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for frag in fragments:
            f.write(json.dumps(asdict(frag), ensure_ascii=False) + "\n")


def load_fragments_jsonl(path: Path) -> list[Fragment]:
    fragments = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                fragments.append(Fragment(**json.loads(line)))
    return fragments
