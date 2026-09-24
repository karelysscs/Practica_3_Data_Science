"""Cálculo y registro observable de costo por llamada a LLM (mismo patrón que
la Tarea 1). No importa UI."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class CostEntry:
    timestamp_utc: str
    call_type: str
    model: str
    input_tokens: int
    output_tokens: int
    usd_cost: float
    latency_seconds: float
    pricing_reference_date: str
    context: str


def compute_cost(model: str, input_tokens: int, output_tokens: int, pricing_cfg: dict) -> float:
    rates = pricing_cfg["usd_per_1m_tokens"].get(model)
    if rates is None:
        raise KeyError(f"No hay tarifa registrada para '{model}' en config.yaml")
    cost = (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates.get("output", 0.0)
    return round(cost, 8)


def log_cost_entry(log_path: Path, call_type: str, model: str, input_tokens: int, output_tokens: int,
                    latency_seconds: float, pricing_cfg: dict, context: str = "") -> CostEntry:
    from datetime import datetime, timezone
    usd_cost = compute_cost(model, input_tokens, output_tokens, pricing_cfg)
    entry = CostEntry(
        timestamp_utc=datetime.now(timezone.utc).isoformat(), call_type=call_type, model=model,
        input_tokens=input_tokens, output_tokens=output_tokens, usd_cost=usd_cost,
        latency_seconds=round(latency_seconds, 4), pricing_reference_date=pricing_cfg["reference_date"],
        context=context[:200],
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return entry


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
