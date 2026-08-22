from typing import Dict, List


class EnvironmentHistory:
    def __init__(
        self,
        base_query: str,
        start_info: str,
        memory: List[str],
        history: List[Dict[str, str]] | None = None,
    ) -> None:
        self._cur_query: str = _get_base_query(base_query, start_info, memory)
        self._history: List[Dict[str, str]] = list(history or [])
        self._last_action: str = ""
        self._is_exhausted: bool = False

    def add(self, label: str, value: str) -> None:
        assert label in {"action", "observation", "human_edit"}
        self._history.append({"label": label, "value": value})
        if label == "action":
            if value == self._last_action:
                self._is_exhausted = True
            else:
                self._last_action = value

    def remove_last_step_if_think(self) -> None:
        if len(self._history) < 2:
            return
        if self._history[-2]["value"].startswith("think[") or self._history[-2]["value"].startswith("> think["):
            self._history = self._history[:-2]

    def check_is_exhausted(self) -> bool:
        return self._is_exhausted

    def reset(self) -> None:
        self._history = []

    def as_steps(self) -> List[Dict[str, str]]:
        steps = []
        cur_action = None
        cur_obs = None
        for item in self._history:
            if item["label"] == "action":
                cur_action = str(item.get("value", ""))
            elif item["label"] == "observation":
                cur_obs = str(item.get("value", ""))
                if cur_action is not None:
                    steps.append({"action": cur_action, "observation": cur_obs})
                    cur_action = None
                    cur_obs = None
        return steps

    def __str__(self) -> str:
        return self.to_prompt_text(include_human_edit=True)

    def to_prompt_text(self, *, include_human_edit: bool = False) -> str:
        s = self._cur_query.rstrip("\n") + "\n"
        rendered = []
        for item in self._history:
            label = item.get("label")
            if label == "human_edit" and not include_human_edit:
                continue
            if label == "action":
                rendered.append(f'> {item.get("value","")}')
            elif label == "observation":
                rendered.append(str(item.get("value", "")))
            elif label == "human_edit":
                rendered.append(f'[human edit]: {item.get("value","")}')
        if rendered:
            s += "\n".join(rendered)
        return s


def _get_base_query(base_query: str, start_info: str, memory: List[str]) -> str:
    query = base_query.rstrip()
    if memory:
        query += "\n\nYour memory for the task below:"
        for i, m in enumerate(memory):
            query += f"\nTrial {i}:\n{m.strip()}"
    query += f"\nHere is the task.\n{start_info}"
    return query
