from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import tiktoken
from openai import OpenAI

from llm_client import LLMConfig, make_chat_llm
from utils import batched, read_json, read_text, safe_json_loads


def _sanitize_identifier(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\W+", "_", text)
    return text.strip("_") or "Rule"


def _extract_rule_number(rule_text: str) -> str | None:
    m = re.match(r"\s*Rule\s+(\d+)\s*[:\-]", str(rule_text), flags=re.IGNORECASE)
    return m.group(1) if m else None


def _assign_rule_ids(action: str, rule_texts: list[str]) -> list[dict[str, str]]:
    sanitized_action = _sanitize_identifier(action)
    used: set[str] = set()
    out: list[dict[str, str]] = []
    for text in rule_texts:
        content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:6]
        num = _extract_rule_number(text)
        if num:
            rule_id = f"Rule_{num}_{sanitized_action}_{content_hash}"
        else:
            rule_id = f"Rule_{sanitized_action}_{content_hash}"
        rule_id = _sanitize_identifier(rule_id)
        if rule_id in used:
            suffix = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
            rule_id = f"{rule_id}_{suffix}"
        used.add(rule_id)
        out.append({"id": rule_id, "text": text})
    return out


def _normalize_rules_loaded(loaded: Any) -> dict[str, list[dict[str, str]]]:
    normalized: dict[str, list[dict[str, str]]] = {}
    if not isinstance(loaded, dict):
        return normalized
    for action, rules in loaded.items():
        if not isinstance(rules, list):
            continue
        normalized[str(action)] = []
        for entry in rules:
            if isinstance(entry, dict):
                rule_id = entry.get("id") or entry.get("rule_id")
                rule_text = entry.get("text") or entry.get("rule") or entry.get("rule_text")
                if not rule_text:
                    continue
                if not rule_id:
                    rule_id = _assign_rule_ids(str(action), [str(rule_text)])[0]["id"]
                normalized[str(action)].append({"id": _sanitize_identifier(str(rule_id)), "text": str(rule_text)})
            else:
                rule_text = str(entry)
                rule_id = _assign_rule_ids(str(action), [rule_text])[0]["id"]
                normalized[str(action)].append({"id": rule_id, "text": rule_text})
    return normalized


class RuleMiner:
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

        system_path = os.path.join(self.prompt_dir, "rule_improve_system_textcraft.txt")
        self.system_prompt = read_text(system_path)
        self.query_template = read_text(os.path.join(self.prompt_dir, "rule_improve_query.txt"))

        self.llm = make_chat_llm(client, config=llm_config, system_prompt=self.system_prompt)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.rules: dict[str, list[dict[str, str]]] = {}

    def _count_tokens(self, obj: Any) -> int:
        if isinstance(obj, str):
            text = obj
        else:
            text = json.dumps(obj, ensure_ascii=False)
        return len(self.tokenizer.encode(text))

    def _truncate_transitions(self, transitions: list[dict[str, Any]], *, max_tokens: int = 5000) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        used = 0
        for t in transitions:
            tok = self._count_tokens(t)
            if used + tok > max_tokens:
                break
            out.append(t)
            used += tok
        return out

    def get_rules_update(self, action_name: str, transitions: list[dict[str, Any]]) -> list[str]:
        existing_rules = self.rules.get(action_name, [])
        existing_texts = [r.get("text", "") for r in existing_rules if isinstance(r, dict)]

        prompt = self.query_template.format(transitions=transitions, rules=existing_texts)
        prompt += f"\n\nMining rules for action: {action_name}\n"

        raw = self.llm(prompt, stop=None)
        parsed = safe_json_loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Rule miner returned non-dict JSON")
        final_rules = parsed.get("final_rules") or []
        if not isinstance(final_rules, list):
            raise ValueError("Rule miner JSON missing final_rules list")
        rule_texts = [str(r).strip() for r in final_rules if str(r).strip()]
        self.rules[action_name] = _assign_rule_ids(action_name, rule_texts)
        return rule_texts

    def get_rules_all(self) -> None:
        buf_pos_path = os.path.join(self.fact_dir, "buffer_correct_temp.json")
        buf_neg_path = os.path.join(self.fact_dir, "buffer_wrong_temp.json")
        buffer_pos = read_json(buf_pos_path) if Path(buf_pos_path).exists() else {}
        buffer_neg = read_json(buf_neg_path) if Path(buf_neg_path).exists() else {}

        rules_file = os.path.join(self.rules_dir, "rules_natural_language.json")
        if Path(rules_file).exists():
            try:
                self.rules = _normalize_rules_loaded(read_json(rules_file))
                print(f"[RuleMiner] Loaded {sum(len(v) for v in self.rules.values())} existing rules from file.")
            except Exception as e:
                print(f"[RuleMiner] Failed to load existing rules: {type(e).__name__}: {e}")
                self.rules = {}

        os.makedirs(self.rules_dir, exist_ok=True)

        # 按动作类型遍历数据
        for act_name, neg in (buffer_neg or {}).items():
            if not isinstance(neg, list) or not neg:
                continue
            pos = buffer_pos.get(act_name, [])
            merged = list(neg) + (list(pos) if isinstance(pos, list) else [])

            # Batch transitions to avoid overlong prompts; update incrementally.
            # bacth_size是20
            for chunk in batched(merged, 20):
                self.get_rules_update(str(act_name), list(chunk))

        with open(rules_file, "w", encoding="utf-8") as f:
            json.dump(self.rules, f, ensure_ascii=False, indent=2)

