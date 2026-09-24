"""Cliente de LLM con proveedor intercambiable (OpenAI o Gemini).

El issue no exige un proveedor específico para el LLM que responde preguntas
(Fase 3) — solo exige OpenAI `text-embedding-3-small` para la comparación de
embeddings de la Fase 4, que no pasa por este módulo. Gemini se agrega como
alternativa gratuita para poder demostrar respuestas reales del motor sin
depender de crédito de OpenAI.

Selección: config.yaml -> rag_engine.llm_provider ("openai" | "gemini").
No importa UI.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float


def call_llm(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
) -> LLMResponse:
    if provider == "openai":
        return _call_openai(model, system_prompt, user_prompt, api_key)
    elif provider == "gemini":
        return _call_gemini(model, system_prompt, user_prompt, api_key)
    raise ValueError(f"Proveedor de LLM no soportado: {provider!r} (usar 'openai' o 'gemini')")


def _call_openai(model: str, system_prompt: str, user_prompt: str, api_key: str) -> LLMResponse:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    latency = time.perf_counter() - t0
    return LLMResponse(
        text=resp.choices[0].message.content,
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
        latency_seconds=latency,
    )


def _call_gemini(model: str, system_prompt: str, user_prompt: str, api_key: str, max_retries: int = 3) -> LLMResponse:
    import requests

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.0},
    }
    t0 = time.perf_counter()
    resp = None
    for attempt in range(max_retries):
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            break
        if resp.status_code in (429, 503) and attempt < max_retries - 1:
            time.sleep(2.0 * (attempt + 1))  # 503/429 = sobrecarga temporal del lado de Google, reintentar
            continue
        break
    latency = time.perf_counter() - t0
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    usage = data.get("usageMetadata", {})
    return LLMResponse(
        text=text,
        input_tokens=usage.get("promptTokenCount", 0),
        output_tokens=usage.get("candidatesTokenCount", 0),
        latency_seconds=latency,
    )
