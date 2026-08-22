from __future__ import annotations

from typing import Any, Dict, List, Optional

from env_history import EnvironmentHistory


def render_env_history_for_buffer(env_history: EnvironmentHistory) -> str:
    """Render EnvironmentHistory while stripping imagination-only rule feedback blocks."""
    base_query = str(getattr(env_history, "_cur_query", "")).rstrip("\n")
    history = list(getattr(env_history, "_history", []))

    filtered: List[Dict[str, str]] = []
    i = 0
    while i < len(history):
        item = history[i]
        label = item.get("label")
        value = str(item.get("value", ""))

        if label == "action" and value.startswith("Action in Imagination"):
            i += 1
            if i < len(history) and history[i].get("label") == "observation":
                i += 1
            continue

        filtered.append(item)
        i += 1

    lines: List[str] = []
    if base_query:
        lines.append(base_query)
    for item in filtered:
        label = item.get("label")
        value = str(item.get("value", ""))
        if label == "action":
            lines.append(f"> {value}")
        elif label == "observation":
            lines.append(value)
        elif label == "human_edit":
            lines.append(f"[human edit]: {value}")
    return "\n".join(lines)


def parse_env_history_steps(env_history_text: str) -> Optional[Dict[str, Any]]:
    """Parse our `EnvironmentHistory.__str__` format into (task, trajectory, actions)."""
    text = (env_history_text or "").replace("\r\n", "\n").replace("\r", "\n")
    task_marker = "Here is the task:"
    idx = text.find(task_marker)
    if idx == -1:
        task_marker = "Here is the task."
        idx = text.find(task_marker)
    if idx == -1:
        return None
    # 去掉Here is the task. 前面的内容
    after = text[idx + len(task_marker) :].strip()
    lines = after.splitlines()

    task_lines: List[str] = []
    trajectory: List[Dict[str, str]] = []
    current_action: Optional[str] = None
    current_obs: List[str] = []

    # 遍历每一行文本
    for line in lines:
        # 如果以>开头，说明是action
        if line.strip().startswith(">"):
            if current_action is not None:
                trajectory.append({"action": current_action, "observation": "\n".join(current_obs).strip()})
            current_action = line.strip()[1:].strip()
            current_obs = []
        else:
            if current_action is None:
                if line.strip():
                    task_lines.append(line.strip())
            else:
                current_obs.append(line)

    if current_action is not None:
        trajectory.append({"action": current_action, "observation": "\n".join(current_obs).strip()})

    task_text = "\n".join(task_lines).strip()
    actions = [step["action"] for step in trajectory]
    # 返回一个字典，包含task、trajectory（一个列表，每一项是一个动作观察对）和actions
    return {"task": task_text, "trajectory": trajectory, "actions": actions}

