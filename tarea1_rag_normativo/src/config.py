"""Carga de configuración (config.yaml) y variables de entorno (.env).

Módulo compartido por todo el pipeline. No debe depender de Streamlit
ni de ninguna librería de UI (regla de arquitectura del engine).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Raíz del proyecto de la Tarea 1 (donde vive config.yaml)
TASK_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = TASK_ROOT.parent

_CONFIG_CACHE: dict[str, Any] | None = None


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Carga config.yaml (con cache en memoria para no releer en cada llamada)."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE

    path = Path(config_path) if config_path else TASK_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if config_path is None:
        _CONFIG_CACHE = cfg
    return cfg


def load_environment() -> None:
    """Carga el archivo .env de la raíz del repo (credenciales, nunca en config.yaml)."""
    env_path = REPO_ROOT / ".env"
    load_dotenv(dotenv_path=env_path, override=False)


def resolve_path(relative: str | Path) -> Path:
    """Resuelve una ruta relativa declarada en config.yaml contra TASK_ROOT."""
    return TASK_ROOT / relative


def get_openai_api_key() -> str | None:
    load_environment()
    return os.environ.get("OPENAI_API_KEY") or None


def get_openai_chat_model(cfg: dict[str, Any] | None = None) -> str:
    load_environment()
    cfg = cfg or load_config()
    env_var = cfg["rag_engine"]["llm_model_env_var"]
    return os.environ.get(env_var, "gpt-4o-mini")


def get_openai_embedding_model(cfg: dict[str, Any] | None = None) -> str:
    load_environment()
    cfg = cfg or load_config()
    return os.environ.get("OPENAI_EMBEDDING_MODEL", cfg["embeddings"]["openai"]["model_name"])
