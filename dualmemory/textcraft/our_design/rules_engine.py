from __future__ import annotations

import json
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType
from typing import Any, Callable


def _extract_func_name(func_str: str) -> str | None:
    m = re.search(r"def\s+(\w+)\s*\(", func_str)
    return m.group(1) if m else None


@dataclass
class RuleCheckResult:
    success: bool
    feedback: str = ""
    suggestion: str = ""
    rule_id: str = ""


class RulesEngine:
    def __init__(self) -> None:
        self._functions: list[Callable[..., Any]] = []

    @property
    def functions(self) -> list[Callable[..., Any]]:
        return list(self._functions)

    def load_code_file(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            self._functions = []
            return
        items = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            raise ValueError(f"Rule code file must contain a JSON list, got: {type(items).__name__}")

        namespace: dict[str, Any] = {}
        functions: list[Callable[..., Any]] = []
        for code in items:
            if not isinstance(code, str):
                continue
            try:
                exec(code, namespace)  # noqa: S102 (intentional: generated rule code)
            except Exception as e:
                name = _extract_func_name(code) or "<unknown>"
                print(f"[RulesEngine] Skipping rule due to exec error: {name}: {type(e).__name__}: {e}")
                continue
            name = _extract_func_name(code)
            if not name:
                continue
            fn = namespace.get(name)
            if isinstance(fn, FunctionType):
                functions.append(fn)
        self._functions = functions

    def check(
        self,
        *,
        state: dict[str, Any] | None,
        action: dict[str, Any] | None,
    ) -> RuleCheckResult:
        if not self._functions:
            return RuleCheckResult(success=True)

        state = state or {}
        action = action or {}

        # 遍历规则函数
        for fn in self._functions:
            try:
                kwargs: dict[str, Any] = {"state": state, "action": action}
                sig = None
                try:
                    sig = inspect.signature(fn)
                except Exception:
                    sig = None
                if sig is not None:
                    if "scene_graph" in sig.parameters:
                        # Backward-compat: older rules used `scene_graph`, but TextCraft no longer provides it.
                        kwargs["scene_graph"] = {}
                    elif "context" in sig.parameters:
                        # Backward-compat: older rules used `context`.
                        kwargs["context"] = {}
                out = fn(**kwargs)
            except Exception as e:
                print(f"[RulesEngine] Rule error: {getattr(fn,'__name__','<rule>')}: {type(e).__name__}: {e}")
                continue

            if not isinstance(out, (tuple, list)) or len(out) != 3:
                # If rule doesn't follow protocol, skip it.
                continue
            feedback, success, suggestion = out
            if success is False:
                return RuleCheckResult(
                    success=False,
                    feedback=str(feedback or "").strip(),
                    suggestion=str(suggestion or "").strip(),
                    rule_id=getattr(fn, "__name__", "") or "",
                )

        return RuleCheckResult(success=True)
