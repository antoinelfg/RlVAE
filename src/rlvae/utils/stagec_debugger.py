"""
Stage C Debugger
================

Centralized helper to collect structured diagnostics for Stage‑C runs.
Instrumentation is enabled when the environment variable
``RLVAE_STAGEC_AUDIT`` is set to ``"1"``.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _try_import_wandb():
    try:
        import wandb  # type: ignore

        return wandb
    except Exception:
        return None


def _serialize_value(value: Any, *, max_list: int = 8) -> Any:
    """Coerce tensors and other rich objects to JSON‑safe summaries."""
    try:
        import torch
    except Exception:
        torch = None  # type: ignore

    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return {"type": "tensor", "shape": list(value.shape), "empty": True}
        with torch.no_grad():
            summary = {
                "type": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "mean": float(value.float().mean().item()),
                "std": float(value.float().std(unbiased=False).item()),
                "min": float(value.min().item()),
                "max": float(value.max().item()),
            }
        return summary
    if isinstance(value, (int, float, str)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > max_list:
            subset = list(value)[:max_list]
            return {
                "type": type(value).__name__,
                "size": len(value),
                "preview": [_serialize_value(v) for v in subset],
            }
        return [_serialize_value(v) for v in value]
    return repr(value)


@dataclass
class StageCDebugger:
    enabled: bool = field(default_factory=lambda: os.environ.get("RLVAE_STAGEC_AUDIT", "0") == "1")
    use_wandb: bool = field(
        default_factory=lambda: os.environ.get("RLVAE_STAGEC_WANDB", "0") == "1"
    )
    print_payloads: bool = field(
        default_factory=lambda: os.environ.get("RLVAE_STAGEC_VERBOSE", "1") == "1"
    )

    def __post_init__(self) -> None:
        self._events: list[Dict[str, Any]] = []
        self._wandb = None
        if self.enabled and self.use_wandb:
            self._wandb = _try_import_wandb()
            if self._wandb is None:
                self.use_wandb = False

    def log_event(
        self,
        name: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        level: str = "INFO",
        flush: bool = False,
    ) -> None:
        if not self.enabled:
            return
        timestamp = time.time()
        record = {
            "time": timestamp,
            "name": name,
            "level": level,
            "payload": _serialize_value(payload or {}),
        }
        self._events.append(record)
        if self.print_payloads:
            print(f"[STAGEC DEBUG] {json.dumps(record, ensure_ascii=False)}")
        if self.use_wandb and self._wandb is not None and getattr(self._wandb, "run", None):
            try:
                self._wandb.log({f"stagec/{name}": record}, commit=False)
            except Exception:
                pass
        if flush:
            self.flush()

    def log_fallback(
        self,
        name: str,
        *,
        reason: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        details = payload or {}
        details.update({"reason": reason})
        self.log_event(name, details, level="FALLBACK")

    def flush(self) -> None:
        if not self.enabled or not self._events:
            return
        if self.print_payloads:
            print(f"[STAGEC DEBUG] Flushing {len(self._events)} events")
        self._events.clear()


_STAGEC_DEBUGGER: Optional[StageCDebugger] = None


def get_stagec_debugger() -> StageCDebugger:
    global _STAGEC_DEBUGGER
    if _STAGEC_DEBUGGER is None:
        _STAGEC_DEBUGGER = StageCDebugger()
    return _STAGEC_DEBUGGER


stagec_debugger = get_stagec_debugger()

