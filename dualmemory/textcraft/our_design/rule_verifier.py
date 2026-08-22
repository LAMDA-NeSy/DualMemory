from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType
from typing import Any, Callable

from openai import OpenAI

from llm_client import LLMConfig, make_chat_llm
from utils import read_json, read_text, safe_json_loads, write_json


def _sanitize_identifier(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\W+", "_", text).strip("_")
    if not text:
        return "Rule"
    if not re.match(r"^[A-Za-z_]", text):
        text = f"Rule_{text}"
    return text


def _extract_func_name(code: str) -> str | None:
    m = re.search(r"def\s+(\w+)\s*\(", code)
    return m.group(1) if m else None


def _extract_rule_id_header(code: str) -> str | None:
    m = re.search(r"^\s*#\s*rule_id\s*=\s*([A-Za-z0-9_]+)\s*$", str(code or ""), flags=re.MULTILINE)
    return _sanitize_identifier(m.group(1)) if m else None


def _derive_rule_id(action: str, rule_text: str, rule_id: str | None = None) -> str:
    sanitized_action = _sanitize_identifier(action)
    if rule_id:
        return _sanitize_identifier(rule_id)

    num = None
    m = re.match(r"\s*Rule\s+(\d+)\s*[:\-]", str(rule_text), flags=re.IGNORECASE)
    if m:
        num = m.group(1)
    if num:
        code_name = f"Rule_{num}_{sanitized_action}"
    else:
        digest = hashlib.md5(str(rule_text).encode("utf-8")).hexdigest()[:8]
        code_name = f"Rule_{digest}_{sanitized_action}"
    return _sanitize_identifier(code_name)


@dataclass
class VerificationStats:
    total_rules: int
    valid_rules: int
    failed_rules: int


class RuleVerifier:
    def __init__(
        self,
        *,
        io_dir: str,
        env_name: str = "textcraft",
        client: OpenAI,
        llm_config: LLMConfig,
    ) -> None:
        self.io_dir = io_dir
        self.env_name = env_name
        self.rules_dir = os.path.join(io_dir, "symbolic_knowledge", env_name)
        self.fact_dir = os.path.join(io_dir, "traj_data", env_name)
        self.prompt_dir = os.path.join(io_dir, "prompts")

        self.system_prompt = read_text(os.path.join(self.prompt_dir, "rule_code_gen_system_textcraft.txt"))
        self.query_template = read_text(os.path.join(self.prompt_dir, "rule_code_gen_query.txt"))

        self.llm = make_chat_llm(client, config=llm_config, system_prompt=self.system_prompt, default_stop=None)

        self.functions: list[Callable[..., Any]] = []
        self.code_strings: list[str] = []

    def load_functions_from_code_strings(self, code_strings: list[str]) -> None:
        namespace: dict[str, Any] = {}
        functions: list[Callable[..., Any]] = []
        for code in code_strings:
            if not isinstance(code, str):
                continue
            try:
                exec(code, namespace)  # noqa: S102 (intentional: generated rule code)
            except Exception as e:
                name = _extract_func_name(code) or "<unknown>"
                print(f"[RuleVerifier] Exec error: {name}: {type(e).__name__}: {e}")
                continue
            name = _extract_func_name(code)
            if not name:
                continue
            fn = namespace.get(name)
            if isinstance(fn, FunctionType):
                functions.append(fn)
        self.functions = functions
        self.code_strings = list(code_strings)

    def load_functions(self) -> None:
        code_file = os.path.join(self.rules_dir, "rules_code.json")
        if not Path(code_file).exists():
            self.functions = []
            self.code_strings = []
            return
        items = read_json(code_file)
        if not isinstance(items, list):
            raise ValueError("rules_code.json must be a JSON list of code strings")
        self.load_functions_from_code_strings([str(s) for s in items if isinstance(s, str)])

    def rule_code_gen(self, action: str, rule_text: str, *, rule_id: str | None = None) -> str:
        code_name = _derive_rule_id(action, rule_text, rule_id)

        prompt = self.query_template.format(rule=rule_text)
        raw = self.llm(prompt, stop=None)

        code = str(raw).strip()
        # Remove accidental markdown fences.
        code = code.replace("```python", "").replace("```", "").strip()
        # Replace function name.
        code = code.replace("expected_rule_code", code_name)
        # Add a stable header for alignment.
        code = f"# rule_id={code_name}\n# {rule_text}\n" + code.strip() + "\n"
        return code

    def rules_code_all(self) -> None:
        rules_file = os.path.join(self.rules_dir, "rules_natural_language.json")
        rule_set = read_json(rules_file)
        if not isinstance(rule_set, dict):
            raise ValueError("rules_natural_language.json must be a JSON object")

        generated: list[str] = []
        for action, rules in rule_set.items():
            if not isinstance(rules, list):
                continue
            for rule in rules:
                if isinstance(rule, dict):
                    rule_id = rule.get("id") or rule.get("rule_id")
                    rule_text = rule.get("text") or rule.get("rule") or rule.get("rule_text")
                else:
                    rule_id = None
                    rule_text = rule
                if not rule_text:
                    continue
                code = self.rule_code_gen(str(action), str(rule_text), rule_id=str(rule_id) if rule_id else None)
                generated.append(code)

        os.makedirs(self.rules_dir, exist_ok=True)
        write_json(os.path.join(self.rules_dir, "rules_code.json"), generated, indent=2)
        self.load_functions()

    def _run_rule(
        self,
        fn: Callable[..., Any],
        *,
        state: dict[str, Any],
        action: dict[str, Any],
    ) -> tuple[str, bool, str]:
        kwargs: dict[str, Any] = {"state": state or {}, "action": action or {}}
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
        if not isinstance(out, (tuple, list)) or len(out) != 3:
            return "", True, ""
        feedback, success, suggestion = out
        return str(feedback or ""), bool(success), str(suggestion or "")

    def functions_verification(self) -> VerificationStats:
        """
        Verify:
        - Positive set (correct actions): any rule that blocks a success is eliminated.
        - Negative set (failed actions): measure coverage (true negatives).
        """
        if not self.functions:
            self.load_functions()

        record: dict[str, list[str]] = {"whole_set": []}
        for fn in self.functions:
            record[fn.__name__] = []

        # 读取正负样本
        pos_path = os.path.join(self.fact_dir, "buffer_correct_all.json")
        neg_path = os.path.join(self.fact_dir, "buffer_wrong_all.json")
        pos = read_json(pos_path) if Path(pos_path).exists() else {}
        neg = read_json(neg_path) if Path(neg_path).exists() else {}

        # 1) positives: eliminate false positives
        # 用正样本排查误杀
        # 遍历动作
        for _act, transitions in (pos or {}).items():
            if not isinstance(transitions, list):
                continue
            # 遍历transition
            for idx, t in enumerate(transitions):
                state = t.get("initial_state") or {}
                action = t.get("action") or {}
                if not isinstance(state, dict) or not isinstance(action, dict):
                    continue
                # 用rule codes测
                for fn in self.functions:
                    name = fn.__name__
                    try:
                        _feedback, ok, _sug = self._run_rule(fn, state=state, action=action)
                    except Exception as e:
                        print(f"[verify+][error] {name} on {_act}_{idx}: {type(e).__name__}: {e}")
                        traceback.print_exc()
                        record[name].append("failed")
                        continue
                    # 误杀就Failed
                    if ok is False:
                        record[name].append("failed")
                    else:
                        record[name].append("0")

        # 2) negatives: coverage of true negatives
        # 用负样本评估覆盖率
        # 遍历动作
        for act, transitions in (neg or {}).items():
            if not isinstance(transitions, list):
                continue
            # 遍历transition
            for idx, t in enumerate(transitions):
                case_id = f"{act}_{idx}"
                record["whole_set"].append(case_id)
                state = t.get("initial_state") or {}
                action = t.get("action") or {}
                if not isinstance(state, dict) or not isinstance(action, dict):
                    continue
                for fn in self.functions:
                    name = fn.__name__
                    if "failed" in record.get(name, []):
                        # Already failed on positives.
                        record[name].append("0")
                        continue
                    try:
                        _feedback, ok, _sug = self._run_rule(fn, state=state, action=action)
                    except Exception as e:
                        print(f"[verify-][error] {name} on {case_id}: {type(e).__name__}: {e}")
                        record[name].append("failed")
                        continue
                    if ok is False:
                        record[name].append(case_id)
                    else:
                        record[name].append("0")

        write_json(os.path.join(self.rules_dir, "verification_result.json"), record, indent=2)

        failed = {k for k, v in record.items() if k != "whole_set" and "failed" in v}
        valid = {k for k in record.keys() if k != "whole_set"} - failed
        return VerificationStats(total_rules=len(record) - 1, valid_rules=len(valid), failed_rules=len(failed))

    def cleanup_unverified_rules(self) -> tuple[set[str], set[str]]:
        verification_path = os.path.join(self.rules_dir, "verification_result.json")
        verification = read_json(verification_path)
        if not isinstance(verification, dict):
            raise ValueError("verification_result.json must be a JSON object")

        failed_rule_names: set[str] = set()
        valid_rule_names: set[str] = set()
        for rule_name, results in verification.items():
            if rule_name == "whole_set" or not isinstance(results, list):
                continue
            normalized_name = _sanitize_identifier(rule_name)
            if "failed" in results:
                failed_rule_names.add(normalized_name)
                continue
            if any(str(item) not in {"0", "00", ""} for item in results):
                valid_rule_names.add(normalized_name)

        rules_nl_path = os.path.join(self.rules_dir, "rules_natural_language.json")
        rules_nl = read_json(rules_nl_path) if Path(rules_nl_path).exists() else {}
        if not isinstance(rules_nl, dict):
            raise ValueError("rules_natural_language.json must be a JSON object")

        cleaned_rules_nl: dict[str, list[dict[str, str]]] = {}
        kept_nl = 0
        removed_nl = 0
        for action, rules in rules_nl.items():
            if not isinstance(rules, list):
                continue
            kept_entries: list[dict[str, str]] = []
            for entry in rules:
                if isinstance(entry, dict):
                    rule_id = entry.get("id") or entry.get("rule_id")
                    rule_text = entry.get("text") or entry.get("rule") or entry.get("rule_text")
                else:
                    rule_id = None
                    rule_text = entry
                if not rule_text:
                    removed_nl += 1
                    continue
                normalized_rule_id = _derive_rule_id(str(action), str(rule_text), str(rule_id) if rule_id else None)
                if normalized_rule_id and normalized_rule_id in valid_rule_names:
                    kept_entries.append({"id": normalized_rule_id, "text": str(rule_text)})
                    kept_nl += 1
                else:
                    removed_nl += 1
            cleaned_rules_nl[str(action)] = kept_entries
        write_json(rules_nl_path, cleaned_rules_nl, indent=2)

        rules_code_path = os.path.join(self.rules_dir, "rules_code.json")
        cleaned_rules_code: list[str] = []
        kept_code = 0
        removed_code = 0
        for code in self.code_strings:
            if not isinstance(code, str):
                continue
            rule_name = _extract_rule_id_header(code) or _extract_func_name(code)
            normalized_rule_name = _sanitize_identifier(rule_name) if rule_name else None
            if normalized_rule_name and normalized_rule_name in valid_rule_names:
                cleaned_rules_code.append(code)
                kept_code += 1
            else:
                removed_code += 1
        write_json(rules_code_path, cleaned_rules_code, indent=2)
        self.load_functions_from_code_strings(cleaned_rules_code)

        print(
            f"[RuleVerifier][cleanup] kept valid={len(valid_rule_names)} failed={len(failed_rule_names)} "
            f"nl_kept={kept_nl} nl_removed={removed_nl} code_kept={kept_code} code_removed={removed_code}"
        )
        return valid_rule_names, failed_rule_names

    def select_rules(self) -> list[str]:
        path = os.path.join(self.rules_dir, "verification_result.json")
        data = read_json(path)
        whole_set = set(data.get("whole_set") or [])
        uncovered = set(whole_set)

        all_rules: dict[str, set[str]] = {}
        for name, items in data.items():
            if name == "whole_set":
                continue
            if not isinstance(items, list):
                continue
            if "failed" in items:
                continue
            covered = {str(x) for x in items if str(x) not in {"0", "00", ""}}
            all_rules[name] = covered

        selected: list[str] = []
        while uncovered:
            best_rule = None
            best_cov: set[str] = set()
            for rule, items in all_rules.items():
                cov = items & uncovered
                if len(cov) > len(best_cov):
                    best_rule = rule
                    best_cov = cov
            if not best_rule or not best_cov:
                break
            selected.append(best_rule)
            uncovered -= best_cov

        sorted_rules = sorted(all_rules.items(), key=lambda kv: len(kv[1]), reverse=True)
        write_json(
            os.path.join(self.rules_dir, "selected_rules.json"),
            {"selected_rules": selected, "sorted_rules": [(k, sorted(list(v))) for k, v in sorted_rules]},
            indent=2,
        )

        # Prune code strings to selected.
        code_by_name: dict[str, str] = {}
        for code in self.code_strings:
            name = _extract_func_name(code) or ""
            if name:
                code_by_name[name] = code

        pruned = [code_by_name[name] for name in selected if name in code_by_name]
        write_json(os.path.join(self.rules_dir, "pruned_rules_code.json"), pruned, indent=2)

        # Also keep a sorted list by coverage (useful for inspection).
        sorted_code: list[str] = []
        for name, _cov in sorted_rules:
            if name in code_by_name:
                sorted_code.append(code_by_name[name])
        write_json(os.path.join(self.rules_dir, "rules_code_sorted.json"), sorted_code, indent=2)

        self.cleanup_unverified_rules()

        return selected
