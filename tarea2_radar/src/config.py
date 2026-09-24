"""Carga de configuración (config.yaml) y variables de entorno (.env) para la Tarea 2.

No importa Streamlit ni ninguna librería de UI.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

TASK_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = TASK_ROOT.parent

_CONFIG_CACHE: dict[str, Any] | None = None


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
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
    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)


def resolve_path(relative: str | Path) -> Path:
    return TASK_ROOT / relative


def get_openai_api_key() -> str | None:
    load_environment()
    return os.environ.get("OPENAI_API_KEY") or None


def get_openai_chat_model(cfg: dict[str, Any] | None = None) -> str:
    load_environment()
    cfg = cfg or load_config()
    env_var = cfg["rag_engine"]["llm_model_env_var"]
    return os.environ.get(env_var, "gpt-4o-mini")


def get_gemini_api_key() -> str | None:
    load_environment()
    return os.environ.get("GEMINI_API_KEY") or None


def get_gemini_chat_model() -> str:
    load_environment()
    return os.environ.get("GEMINI_CHAT_MODEL", "gemini-3.6-flash")


def get_llm_provider(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_config()
    return cfg["rag_engine"].get("llm_provider", "openai")


def get_llm_credentials(cfg: dict[str, Any] | None = None) -> tuple[str, str, str | None]:
    """Devuelve (provider, model, api_key) según rag_engine.llm_provider en config.yaml."""
    cfg = cfg or load_config()
    provider = get_llm_provider(cfg)
    if provider == "gemini":
        return "gemini", get_gemini_chat_model(), get_gemini_api_key()
    return "openai", get_openai_chat_model(cfg), get_openai_api_key()
