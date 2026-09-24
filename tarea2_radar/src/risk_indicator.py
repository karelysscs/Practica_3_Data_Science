"""Fase 5 — Indicador de riesgo: proporción de adjudicaciones con un solo
postor (single-bidder awards), por departamento y por comprador.

Fundamento: la Open Contracting Partnership (OCP) documenta la proporción de
procesos con un único postor como una de las "red flags" (banderas rojas) de
integridad en contrataciones públicas más citadas en su literatura de
indicadores (ver: Open Contracting Partnership — "Red Flags for Integrity",
https://www.open-contracting.org/, y el "Procurement Red Flags" toolkit del
Government Transparency Institute usado por la OCP): un solo postor reduce la
presión competitiva sobre el precio y es un proxy usual de riesgo de colusión
o direccionamiento, aunque no es prueba concluyente por sí solo.

No importa UI.
"""
from __future__ import annotations

import pandas as pd


def _awarded_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_single_bidder_award"].notna()]


def single_bidder_share_by_department(df: pd.DataFrame) -> pd.DataFrame:
    awarded = _awarded_only(df)
    awarded = awarded[awarded["department_is_valid"]]  # solo los 25 departamentos válidos
    grouped = awarded.groupby("buyer_department").agg(
        n_awards=("ocid", "count"),
        n_single_bidder=("is_single_bidder_award", "sum"),
    )
    grouped["single_bidder_share"] = (grouped["n_single_bidder"] / grouped["n_awards"]).round(4)
    return grouped.sort_values("single_bidder_share", ascending=False).reset_index()


def top_buyers_by_single_bidder_share(df: pd.DataFrame, min_processes: int, top_n: int) -> pd.DataFrame:
    """Ranking de compradores con mayor proporción de adjudicaciones a un solo
    postor, exigiendo un mínimo de procesos adjudicados (`min_processes`) para
    evitar que un comprador con 1 solo proceso (100% o 0%) distorsione el
    ranking — umbral justificado: por debajo de ~5 procesos, el ratio no es
    estadísticamente representativo (un solo caso mueve el share en 20-100
    puntos porcentuales)."""
    awarded = _awarded_only(df)
    grouped = awarded.groupby(["buyer_id", "buyer_name", "buyer_department"]).agg(
        n_awards=("ocid", "count"),
        n_single_bidder=("is_single_bidder_award", "sum"),
    )
    grouped = grouped[grouped["n_awards"] >= min_processes]
    grouped["single_bidder_share"] = (grouped["n_single_bidder"] / grouped["n_awards"]).round(4)
    return grouped.sort_values(
        ["single_bidder_share", "n_awards"], ascending=[False, False]
    ).head(top_n).reset_index()
