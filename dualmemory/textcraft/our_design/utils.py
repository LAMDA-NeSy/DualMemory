from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_JSON_BLOCK_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def default_io_dir() -> str:
    env_dir = os.environ.get("DUALMEMORY_TEXTCRAFT_DIR")
    if env_dir:
        return str(Path(env_dir).expanduser().resolve())
    return str(Path(__file__).resolve().parent)


def ensure_dir(path: str | Path) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(read_text(path))


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")


def list_traj_ids(traj_root: str | Path) -> list[int]:
    root = Path(traj_root)
    if not root.exists():
        return []
    ids: list[int] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("traj_"):
            continue
        try:
            ids.append(int(name.split("traj_", 1)[1]))
        except ValueError:
            continue
    return sorted(ids)


def extract_json_block(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1)
    return text


def safe_json_loads(text: str) -> Any:
    text = extract_json_block(text)
    text = text.strip()
    if not text:
        raise ValueError("empty JSON")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try removing trailing commas.
        fixed = re.sub(r",\s*([}\]])", r"\1", text)
        return json.loads(fixed)


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int = 6
    base_delay_s: float = 1.5
    max_delay_s: float = 30.0


def with_retries(fn, *, retry: RetryConfig, is_retryable=None):
    last_err: Exception | None = None
    for attempt in range(retry.max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if is_retryable is not None and not is_retryable(e):
                raise
            if attempt == retry.max_retries - 1:
                break
            delay = min(retry.max_delay_s, retry.base_delay_s * (2**attempt))
            time.sleep(delay)
    assert last_err is not None
    raise last_err


def compact_json(obj: Any, *, max_chars: int = 4000) -> str:
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    if len(raw) <= max_chars:
        return raw
    return raw[: max(0, max_chars - 3)] + "..."


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
