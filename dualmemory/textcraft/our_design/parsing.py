from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_ACTION_PREFIX_RE = re.compile(r"^(?:action|act)\s*:\s*", re.IGNORECASE)
_THOUGHT_PREFIX_RE = re.compile(r"^(?:thought)\s*:\s*", re.IGNORECASE)

_INV_ITEM_RE = re.compile(r"\[([^\]]+)\]\s*\((\d+)\)")


def first_line(text: str) -> str:
    return (text or "").strip().splitlines()[0].strip()


def parse_action(raw: str) -> str:
    line = first_line(raw).lstrip("> ").strip()
    if not line:
        return ""

    line = _ACTION_PREFIX_RE.sub("", line).strip()
    if _THOUGHT_PREFIX_RE.match(line):
        line = _THOUGHT_PREFIX_RE.sub("think: ", line).strip()

    if line.lower().startswith("fetch "):
        line = "get " + line[len("fetch ") :].strip()

    if line.lower().startswith("inventory"):
        return "inventory"

    return line.rstrip(".").strip()


def is_think(action: str) -> bool:
    return action.lower().startswith("think:")


def is_inventory_action(action: str) -> bool:
    return action.strip().lower() == "inventory"


def is_get_action(action: str) -> bool:
    return action.strip().lower().startswith("get ")


def is_craft_action(action: str) -> bool:
    return action.strip().lower().startswith("craft ")


def normalize_item_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s+", " ", name)
    return name


@dataclass(frozen=True)
class ParsedAction:
    name: str
    args: dict[str, Any]
    raw: str


def parse_get_action(action: str) -> ParsedAction | None:
    text = action.strip()
    m = re.match(r"^get\s+(\d+)\s+(.+)$", text, flags=re.IGNORECASE)
    if not m:
        return None
    count = int(m.group(1))
    item = normalize_item_name(m.group(2))
    return ParsedAction(name="get", args={"count": count, "item": item}, raw=action)


def _split_ingredients(text: str) -> list[str]:
    # Ingredients are comma-separated; items may contain spaces.
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p]


def parse_craft_action(action: str) -> ParsedAction | None:
    text = action.strip()
    m = re.match(r"^craft\s+(\d+)\s+(.+?)\s+using\s+(.+)$", text, flags=re.IGNORECASE)
    if not m:
        return None
    out_count = int(m.group(1))
    out_item = normalize_item_name(m.group(2))
    ing_text = m.group(3).strip()

    inputs: list[dict[str, Any]] = []
    for part in _split_ingredients(ing_text):
        m2 = re.match(r"^(\d+)\s+(.+)$", part)
        if not m2:
            # Keep raw ingredient chunk for debugging, but still return a parsed action.
            inputs.append({"count": None, "item": normalize_item_name(part), "raw": part})
            continue
        inputs.append({"count": int(m2.group(1)), "item": normalize_item_name(m2.group(2))})

    return ParsedAction(
        name="craft",
        args={"count": out_count, "item": out_item, "inputs": inputs},
        raw=action,
    )


def parse_recipe_line(line: str) -> dict[str, Any] | None:
    parsed = parse_craft_action(line)
    if not parsed:
        return None
    return {"output": {"item": parsed.args["item"], "count": parsed.args["count"]}, "inputs": parsed.args["inputs"], "raw": line.strip()}


def parse_goal(problem_text: str) -> dict[str, Any] | None:
    # Examples:
    # "Goal: craft dark oak sign."
    # "Goal: craft 2 oak planks"
    for line in (problem_text or "").splitlines():
        if not line.strip().lower().startswith("goal:"):
            continue
        goal_text = line.split(":", 1)[1].strip()
        m = re.match(r"^craft\s+(\d+)\s+(.+?)[\.\s]*$", goal_text, flags=re.IGNORECASE)
        if m:
            return {"item": normalize_item_name(m.group(2)), "count": int(m.group(1))}
        m = re.match(r"^craft\s+(.+?)[\.\s]*$", goal_text, flags=re.IGNORECASE)
        if m:
            return {"item": normalize_item_name(m.group(1)), "count": 1}
    return None


def parse_recipes(problem_text: str) -> list[dict[str, Any]]:
    recipes: list[dict[str, Any]] = []
    in_block = False
    for line in (problem_text or "").splitlines():
        if line.strip().lower().startswith("crafting commands"):
            in_block = True
            continue
        if in_block and line.strip().lower().startswith("goal:"):
            break
        if not in_block:
            continue
        line = line.strip()
        if not line:
            continue
        if not line.lower().startswith("craft "):
            continue
        parsed = parse_recipe_line(line)
        if parsed:
            recipes.append(parsed)
    return recipes


def parse_inventory_observation(observation: str) -> dict[str, int] | None:
    # Observation: "Inventory: [diorite] (3) [granite] (1)"
    obs = (observation or "").strip()
    if not obs.lower().startswith("inventory:"):
        return None
    items: dict[str, int] = {}
    for item, count in _INV_ITEM_RE.findall(obs):
        items[normalize_item_name(item)] = int(count)
    return items


def infer_action_success(action: str, observation: str) -> bool:
    action = (action or "").strip()
    obs = (observation or "").strip()
    if is_think(action):
        return True
    if is_inventory_action(action):
        return True
    if obs.startswith("Invalid action:"):
        return False
    if obs.startswith("Could not"):
        return False
    if is_get_action(action):
        return obs.startswith("Got ")
    if is_craft_action(action):
        return obs.startswith("Crafted ")
    # Unknown actions are treated as failures (should be blocked upstream).
    return False


def to_action_json(action: str) -> dict[str, Any] | None:
    action = (action or "").strip()
    if not action:
        return None
    if is_inventory_action(action):
        return {"name": "inventory", "args": {}, "raw": action}
    parsed = parse_get_action(action)
    if parsed:
        return {"name": parsed.name, "args": parsed.args, "raw": parsed.raw}
    parsed = parse_craft_action(action)
    if parsed:
        return {"name": parsed.name, "args": parsed.args, "raw": parsed.raw}
    if is_think(action):
        return {"name": "think", "args": {"text": action[len("think:") :].strip()}, "raw": action}
    return {"name": "unknown", "args": {"text": action}, "raw": action}
