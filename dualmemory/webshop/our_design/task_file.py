"""Load and validate the fixed WebShop task order used in the paper runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_webshop_tasks(task_file: str) -> list[dict[str, Any]]:
    path = Path(task_file).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"WebShop task file contains no tasks: {path}")

    tasks: list[dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError(f"Each WebShop task must be an object: {path}")
        task = entry.get("task")
        session_idx = entry.get("session_idx")
        key = entry.get("key")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Each WebShop task must contain non-empty task text: {path}")
        if not isinstance(session_idx, str):
            raise ValueError(f"Each WebShop task must contain session_idx: {path}")
        match = re.fullmatch(r"fixed_(\d+)", session_idx)
        if not match:
            raise ValueError(f"WebShop session_idx must use fixed_<id>: {session_idx!r}")
        if not isinstance(key, dict) or not all(field in key for field in ("asin", "query", "name", "goal_options")):
            raise ValueError(f"Each WebShop task must contain an ExpeL-style key: {path}")

        task_text = re.sub(r"\s*\[Search\]\s*$", "", task).strip()
        tasks.append(
            {
                "task_id": int(match.group(1)),
                "task_text": task_text,
                "session_idx": session_idx,
                "key": key,
            }
        )
    return tasks


def normalize_task_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def validate_webshop_task(expected: str, observed: str) -> None:
    if normalize_task_text(expected) not in normalize_task_text(observed):
        raise RuntimeError(
            "WebShop task mismatch: the server returned a different instruction than "
            "the published task file. Check the StateAct/WebShop data version."
        )
