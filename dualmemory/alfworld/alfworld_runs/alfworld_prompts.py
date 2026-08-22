from __future__ import annotations


def build_action_system_prompt(*, fewshot: str, max_steps: int) -> str:
    fewshot = (fewshot or "").strip()
    prompt = f"""You are an intelligent ALFWorld household assistant.
Your job is to solve the given task as efficiently as possible. You only have {int(max_steps)} steps to do so.
The environment gives you an Observation each turn; you must output exactly ONE line per turn.
If you are thinking or planning, start the line with 'think: '.
If you are acting in the environment, output the action directly (do NOT prefix with '>').
Do not output any extra text.
"""
    if fewshot:
        prompt += f"\nHere are example interactions:\n{fewshot}\n"
    return prompt.strip()

