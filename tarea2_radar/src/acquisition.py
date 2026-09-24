"""Fase 1 — Adquisición de datos del Portal de Contrataciones Abiertas (OECE).

Dos vías, tal como pide el issue:
1) Descarga masiva (bulk) de archivos mensuales JSON — para el histórico
   (config.yaml: source.bulk_months). Re-ejecutable y sin duplicar: si el ZIP ya
   existe localmente, no se vuelve a descargar.
2) API para actualizaciones RECIENTES únicamente (config.yaml: incremental_updates):
   se listan releases recientes (granularidad "release", vía /releases?dateFrom=)
   para descubrir qué OCID tuvieron actividad, y por cada uno se pide su record
   compilado actual (granularidad "record", vía /record/{ocid}) — la misma forma
   que trae cada entrada de los archivos bulk. Esto refleja la distinción real
   del estándar OCDS entre "release" (un evento/publicación puntual) y "record"
   (la vista consolidada y vigente de todas las releases de un proceso).

Throttling: pausa entre llamadas (incremental_updates.throttle_seconds).
Caching: cada /record/{ocid} se cachea en disco con un TTL para no re-pedir lo
mismo en cada corrida (incremental_updates.cache_ttl_hours).

No importa UI.
"""
from __future__ import annotations

import json
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from src.config import load_config, resolve_path


def _session(cfg: dict) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": cfg["http"]["user_agent"]})
    return s


def _get_with_retries(session: requests.Session, url: str, cfg: dict, params: dict | None = None) -> requests.Response:
    last_exc = None
    for attempt in range(cfg["http"]["retries"]):
        try:
            resp = session.get(url, params=params, timeout=cfg["http"]["timeout_seconds"])
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(cfg["http"]["retry_backoff_seconds"] * (attempt + 1))
    raise RuntimeError(f"Falló GET {url} tras {cfg['http']['retries']} intentos: {last_exc}")


# ==========================================================================
# 1) Descarga masiva mensual (bulk)
# ==========================================================================

def download_bulk_month(year: int, month: int, cfg: dict) -> Path:
    """Descarga (si no existe) el ZIP mensual y devuelve la ruta al JSON extraído.
    Re-ejecutable: si el JSON ya existe localmente, no vuelve a descargar ni a
    descomprimir (idempotente)."""
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    src = cfg["source"]
    json_name = f"{year}-{month:02d}_{src['system']}.json"
    json_path = raw_dir / json_name
    if json_path.exists():
        return json_path  # cache: ya descargado y extraído en una corrida anterior

    zip_name = f"{year}-{month:02d}_{src['system']}_json.zip"
    zip_path = raw_dir / zip_name
    url = src["bulk_file_url_template"].format(
        api_base_url=src["api_base_url"], system=src["system"], year=year, month=month
    )

    if not zip_path.exists():
        session = _session(cfg)
        resp = _get_with_retries(session, url, cfg)
        zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(zip_path) as zf:
        # el ZIP contiene un único JSON con el nombre "<year>-<month>_<system>.json"
        names = zf.namelist()
        target = names[0] if len(names) == 1 else next(n for n in names if n.endswith(".json"))
        with zf.open(target) as f_in, open(json_path, "wb") as f_out:
            f_out.write(f_in.read())

    return json_path


def download_all_bulk_months(cfg: dict) -> list[Path]:
    return [download_bulk_month(m["year"], m["month"], cfg) for m in cfg["source"]["bulk_months"]]


# ==========================================================================
# 2) Actualizaciones recientes vía API (release -> record)
# ==========================================================================

def _cache_path(ocid: str, cfg: dict) -> Path:
    cache_dir = resolve_path(cfg["paths"]["raw_dir"]) / "api_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = ocid.replace("/", "_")
    return cache_dir / f"{safe_name}.json"


def discover_recent_ocids(cfg: dict) -> list[str]:
    """Recorre /releases?dateFrom=... (granularidad RELEASE) para descubrir qué
    procesos tuvieron actividad reciente. Con throttling y un techo de páginas."""
    inc = cfg["incremental_updates"]
    since = (datetime.now(timezone.utc) - timedelta(days=inc["lookback_days"])).strftime("%Y-%m-%d")
    session = _session(cfg)
    api_base = cfg["source"]["api_base_url"]

    ocids_seen: list[str] = []
    seen_set = set()
    page = 1
    while page <= inc["max_pages"]:
        resp = _get_with_retries(session, f"{api_base}/releases", cfg, params={"dateFrom": since, "page": page})
        data = resp.json()
        releases = data.get("releases", [])
        if not releases:
            break
        for r in releases:
            ocid = r.get("ocid")
            if ocid and ocid not in seen_set:
                seen_set.add(ocid)
                ocids_seen.append(ocid)
        if not data.get("links", {}).get("next"):
            break
        page += 1
        time.sleep(inc["throttle_seconds"])

    return ocids_seen


def fetch_record(ocid: str, cfg: dict) -> dict | None:
    """Obtiene el record compilado actual de un OCID (granularidad RECORD),
    usando cache en disco con TTL para no repetir la llamada innecesariamente."""
    inc = cfg["incremental_updates"]
    cache_file = _cache_path(ocid, cfg)

    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < inc["cache_ttl_hours"]:
            return json.loads(cache_file.read_text(encoding="utf-8"))

    session = _session(cfg)
    api_base = cfg["source"]["api_base_url"]
    try:
        resp = _get_with_retries(session, f"{api_base}/record/{ocid}", cfg)
    except RuntimeError:
        return None  # OCID no encontrado / error transitorio: se omite, no rompe el batch
    data = resp.json()
    records = data.get("records", [])
    if not records:
        return None
    record = records[0]
    cache_file.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    time.sleep(inc["throttle_seconds"])
    return record


def fetch_recent_updates(cfg: dict) -> list[dict]:
    """Punto de entrada de la Fase 1 (actualizaciones recientes vía API).
    Devuelve una lista de records (misma forma que las entradas de los archivos
    bulk) listos para hacer upsert en el dataset procesado por OCID."""
    ocids = discover_recent_ocids(cfg)
    records = []
    for ocid in ocids:
        rec = fetch_record(ocid, cfg)
        if rec is not None:
            records.append(rec)
    return records
