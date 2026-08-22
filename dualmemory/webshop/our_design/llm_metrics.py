import json
import os
import time
from typing import Any, Dict, Optional


_METRICS: Dict[str, Any] = {
    "llm_call_count": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "latency_seconds": 0.0,
    "calls": [],
}


def reset_metrics() -> None:
    _METRICS["llm_call_count"] = 0
    _METRICS["prompt_tokens"] = 0
    _METRICS["completion_tokens"] = 0
    _METRICS["total_tokens"] = 0
    _METRICS["latency_seconds"] = 0.0
    _METRICS["calls"] = []


def _usage_value(usage: Any, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(key, 0)
    else:
        value = getattr(usage, key, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _extract_usage(response: Any) -> Any:
    usage = getattr(response, "usage", None)
    if usage is not None:
        return usage
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata:
        return {
            "prompt_tokens": _usage_value(usage_metadata, "input_tokens"),
            "completion_tokens": _usage_value(usage_metadata, "output_tokens"),
            "total_tokens": _usage_value(usage_metadata, "total_tokens"),
        }
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
        if token_usage:
            return token_usage
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        return llm_output.get("token_usage") or llm_output.get("usage")
    return None


def record_response(
    *,
    model: str,
    component: str,
    started_at: float,
    response: Any,
) -> None:
    latency = max(0.0, time.perf_counter() - started_at)
    usage = _extract_usage(response)
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens") or prompt_tokens + completion_tokens

    _METRICS["llm_call_count"] += 1
    _METRICS["prompt_tokens"] += prompt_tokens
    _METRICS["completion_tokens"] += completion_tokens
    _METRICS["total_tokens"] += total_tokens
    _METRICS["latency_seconds"] += latency
    _METRICS["calls"].append(
        {
            "component": component,
            "model": model,
            "latency_seconds": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    )


def get_metrics() -> Dict[str, Any]:
    calls = list(_METRICS["calls"])
    count = int(_METRICS["llm_call_count"])
    latency = float(_METRICS["latency_seconds"])
    return {
        "llm_call_count": count,
        "prompt_tokens": int(_METRICS["prompt_tokens"]),
        "completion_tokens": int(_METRICS["completion_tokens"]),
        "total_tokens": int(_METRICS["total_tokens"]),
        "latency_seconds": latency,
        "avg_latency_seconds": latency / count if count else 0.0,
        "calls": calls,
    }


def write_metrics(path: str, *, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = get_metrics()
    if extra:
        payload.update(extra)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
