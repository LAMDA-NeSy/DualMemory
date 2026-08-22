"""Load and validate fixed TextCraft task specifications used in the paper runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_textcraft_tasks(task_file: str) -> list[dict[str, Any]]:
    path = Path(task_file).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"TextCraft task file contains no tasks: {path}")
    tasks: list[dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict) or not isinstance(entry.get("seed"), int):
            raise ValueError(f"Each TextCraft task must contain an integer seed: {path}")
        task = entry.get("task")
        key = entry.get("key")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Each TextCraft task must contain task: {path}")
        if not isinstance(key, dict):
            raise ValueError(f"Each TextCraft task must contain a key: {path}")
        if not isinstance(key.get("goal"), str) or not key["goal"].strip():
            raise ValueError(f"Each TextCraft key must contain goal: {path}")
        recipe_lines = key.get("recipe_lines")
        if not isinstance(recipe_lines, list) or not recipe_lines or not all(
            isinstance(line, str) and line.startswith("craft ") for line in recipe_lines
        ):
            raise ValueError(f"Each TextCraft key must contain recipe_lines: {path}")

        goal_match = re.search(r"\nGoal: craft (.+)\.$", task.strip())
        if not goal_match or goal_match.group(1).strip() != key["goal"].strip():
            raise ValueError(f"TextCraft task/key goal mismatch: {path}")
        task_recipe_lines = [line for line in task.splitlines() if line.startswith("craft ")]
        if task_recipe_lines != recipe_lines:
            raise ValueError(f"TextCraft task/key recipe mismatch: {path}")

        tasks.append({"seed": entry["seed"], "task_text": task.strip(), "key": key})
    return tasks


def normalize_task_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def validate_textcraft_task(expected: dict[str, Any], observed: str) -> None:
    expected_text = str(expected.get("task_text") or "")
    if normalize_task_text(expected_text) != normalize_task_text(observed):
        raise RuntimeError(
            "TextCraft task mismatch: the environment returned a different task than "
            "the published task file. Check the installed TextCraft data version."
        )
    goal_match = re.search(r"\nGoal: craft (.+)\.$", str(observed).strip())
    if not goal_match or goal_match.group(1).strip() != str(expected["key"]["goal"]).strip():
        raise RuntimeError("TextCraft task goal does not match the published task key.")
    observed_recipe_lines = [line for line in str(observed).splitlines() if line.startswith("craft ")]
    if observed_recipe_lines != list(expected["key"]["recipe_lines"]):
        raise RuntimeError("TextCraft recipe set does not match the published task key.")
