from __future__ import annotations
from typing import Dict, List


REACT_ACTION_SYSTEM_PROMPT = (
    "You are an agent controlling WebShop via a strict action API. "
    "Output MUST be exactly ONE line, and it MUST be exactly one of: "
    "search[...], click[...], think[...]"
)


def build_action_system_prompt(*, fewshot: str, max_steps: int) -> str:
    fewshot = (fewshot or "").strip()
    prompt = f"""You are an intelligent WebShop assistant.
Your job is to buy an item that matches the instruction as close as possible. You only have {int(max_steps)} steps to do so.
The environment gives you an ```Observation```, you need to produce the correct output at every turn.
Make sure only to use the functions availble in the observations, follow the syntax of the example.
Output MUST be exactly ONE line, and it MUST be exactly one of: think[...], search[...], click[...]. Do not output any extra text.
"""
    if fewshot:
        prompt += f"\nHere are example interactions:\n{fewshot}\n"
    return prompt.strip()


def build_react_action_system_prompt() -> str:
    return REACT_ACTION_SYSTEM_PROMPT


def build_react_action_user_prompt(
    *,
    init_prompt: str,
    first_observation: str,
    steps: List[Dict[str, str]],
) -> str:
    suffix = f"{first_observation}\n\nAction:"
    for step in steps:
        action = str(step.get("action", "")).strip()
        observation = str(step.get("observation", ""))
        suffix += f" {action}\nObservation: {observation}\n\nAction:"

    init_prompt = init_prompt or ""
    return init_prompt + suffix


def postprocess_action_text(action: str) -> str:
    action = (action or "").strip()
    if action.startswith(">"):
        action = action[1:].strip()
    if action.lower().startswith("action:"):
        action = action.split(":", 1)[1].strip()
    return action.strip()
