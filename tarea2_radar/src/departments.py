"""Fase 2 — Normalización de la ubicación del comprador a los 25 departamentos
del Perú, con reglas documentadas.

Los 25 departamentos oficiales (incluye la Provincia Constitucional del Callao,
tratada como departamento para fines de esta normalización, como hace el propio
INEI). No importa UI.
"""
from __future__ import annotations

import re
import unicodedata

CANONICAL_DEPARTMENTS = [
    "AMAZONAS", "ANCASH", "APURIMAC", "AREQUIPA", "AYACUCHO", "CAJAMARCA",
    "CALLAO", "CUSCO", "HUANCAVELICA", "HUANUCO", "ICA", "JUNIN",
    "LA LIBERTAD", "LAMBAYEQUE", "LIMA", "LORETO", "MADRE DE DIOS",
    "MOQUEGUA", "PASCO", "PIURA", "PUNO", "SAN MARTIN", "TACNA",
    "TUMBES", "UCAYALI",
]
assert len(CANONICAL_DEPARTMENTS) == 25

# Alias conocidos -> canónico. Se documentan explícitamente (regla de "reglas
# documentadas" pedida por el issue) en vez de aplicar una heurística ciega.
_ALIASES = {
    "LIMA METROPOLITANA": "LIMA",
    "PROVINCIA DE LIMA": "LIMA",
    "MUNICIPALIDAD METROPOLITANA DE LIMA": "LIMA",
    "PROVINCIA CONSTITUCIONAL DEL CALLAO": "CALLAO",
    "PROV. CONST. DEL CALLAO": "CALLAO",
    "REGION CALLAO": "CALLAO",
    "SAN MARTÍN": "SAN MARTIN",
    "ANCASH ": "ANCASH",
}


def _strip_accents_upper(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.strip().upper())
    return "".join(c for c in text if not unicodedata.combining(c))


def normalize_department(raw_value: str | None) -> tuple[str | None, bool]:
    """Devuelve (valor_normalizado_o_None, es_valido).

    es_valido=True si el resultado es uno de los 25 departamentos canónicos.
    Si raw_value no mapea a ninguno, se devuelve (raw_value_limpio, False) para
    que quede registrado en el reporte de calidad como "valor no-departamento".
    """
    if not raw_value or not raw_value.strip():
        return None, False

    cleaned = _strip_accents_upper(raw_value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned in CANONICAL_DEPARTMENTS:
        return cleaned, True

    alias_target = _ALIASES.get(cleaned) or _ALIASES.get(raw_value.strip().upper())
    if alias_target:
        return alias_target, True

    # Segundo intento: el valor podría venir con acentos ya mapeados en _ALIASES
    # con acento (ej. "SAN MARTÍN"); ya cubierto arriba por _strip_accents_upper
    # + comparación sin acento contra CANONICAL_DEPARTMENTS.

    return cleaned, False
