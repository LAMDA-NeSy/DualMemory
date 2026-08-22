import hashlib
import json
import os
import random
import re
from typing import Any, Dict, List

from api_config import get_api_model
from json_utils import fix_and_parse_json
from utils import get_openai_client


DEFAULT_RULE_MINER_MODEL = get_api_model("rule_miner", "gpt-4o-mini")


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class RuleMiner:
    def __init__(
        self,
        io_dir: str,
        env_name: str = "webshop",
        model_name: str = DEFAULT_RULE_MINER_MODEL,
        temperature: float = 0,
        batch_size: int = 10,
    ) -> None:
        self.io_dir = io_dir
        self.env_name = env_name
        self.model_name = model_name
        self.temperature = float(temperature)
        self.batch_size = int(batch_size)

        self.rules_dir = os.path.join(io_dir, "symbolic_knowledge", env_name)
        self.prompt_dir = os.path.join(io_dir, "prompts")
        self.fact_dir = os.path.join(io_dir, "traj_data", env_name)
        self.rules: Dict[str, List[Dict[str, str]]] = {}

    def _sanitize_identifier(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\\W+", "_", text)
        return text.strip("_")

    def _extract_rule_number(self, rule_text: str) -> str | None:
        match = re.match(r"\\s*Rule\\s+(\\d+)\\s*[:\\-]", rule_text, flags=re.IGNORECASE)
        return match.group(1) if match else None

    def _normalize_rules_loaded(self, loaded: dict) -> dict:
        normalized: dict = {}
        if not isinstance(loaded, dict):
            return normalized
        for action, rules in loaded.items():
            if not isinstance(rules, list):
                continue
            sanitized_action = self._sanitize_identifier(str(action))
            used_ids: set[str] = set()
            normalized[action] = []
            for entry in rules:
                if isinstance(entry, dict):
                    rule_id = entry.get("id") or entry.get("rule_id")
                    rule_text = entry.get("text") or entry.get("rule") or entry.get("rule_text")
                else:
                    rule_id = None
                    rule_text = str(entry)
                if not rule_text:
                    continue

                if not rule_id:
                    content_hash = hashlib.md5(str(rule_text).encode("utf-8")).hexdigest()[:6]
                    rule_number = self._extract_rule_number(str(rule_text))
                    if rule_number:
                        rule_id = f"Rule_{rule_number}_{sanitized_action}_{content_hash}"
                    else:
                        rule_id = f"Rule_{sanitized_action}_{content_hash}"
                rule_id = self._sanitize_identifier(str(rule_id))
                if rule_id in used_ids:
                    suffix = hashlib.md5(str(rule_text).encode("utf-8")).hexdigest()[:8]
                    rule_id = f"{rule_id}_{suffix}"
                used_ids.add(rule_id)
                normalized[action].append({"id": rule_id, "text": str(rule_text)})
        return normalized

    def _assign_rule_ids(self, action: str, rule_texts: List[str]) -> List[Dict[str, str]]:
        sanitized_action = self._sanitize_identifier(str(action))
        used_ids: set[str] = set()
        out: List[Dict[str, str]] = []
        for rule_text in rule_texts:
            content_hash = hashlib.md5(rule_text.encode("utf-8")).hexdigest()[:6]
            rule_number = self._extract_rule_number(rule_text)
            if rule_number:
                rule_id = f"Rule_{rule_number}_{sanitized_action}_{content_hash}"
            else:
                rule_id = f"Rule_{sanitized_action}_{content_hash}"
            rule_id = self._sanitize_identifier(rule_id)
            if rule_id in used_ids:
                suffix = hashlib.md5(rule_text.encode("utf-8")).hexdigest()[:8]
                rule_id = f"{rule_id}_{suffix}"
            used_ids.add(rule_id)
            out.append({"id": rule_id, "text": rule_text})
        return out

    def _llm_json(self, system: str, prompt: str) -> Dict[str, Any]:
        client = get_openai_client(self.model_name)
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        text = (resp.choices[0].message.content or "").strip()
        parsed = fix_and_parse_json(text)
        return parsed if isinstance(parsed, dict) else {}

    def get_rules_update(self, action_name: str, transitions: List[Dict[str, Any]]) -> List[str]:
        system_path = os.path.join(self.prompt_dir, f"rule_improve_system_{self.env_name}.txt")
        query_path = os.path.join(self.prompt_dir, "rule_improve_query.txt")
        system = _load_text(system_path)
        query_template = _load_text(query_path)

        existing_rules = self.rules.get(action_name, [])
        existing_rule_texts = [
            r["text"] if isinstance(r, dict) and "text" in r else str(r) for r in existing_rules
        ]

        rules_candidate: List[str] = []
        total = len(transitions)
        for i in range(0, total, self.batch_size):
            batch = transitions[i : i + self.batch_size]
            prompt = query_template.format(transitions=batch, rules=existing_rule_texts)
            prompt += f"\nMining rules for action '{action_name}'."
            parsed = self._llm_json(system=system, prompt=prompt)
            final_rules = parsed.get("final_rules") or parsed.get("rules") or []
            if isinstance(final_rules, list):
                rules_candidate.extend([str(r).strip() for r in final_rules if str(r).strip()])

        # De-dup while preserving order.
        # 去重
        seen = set()
        deduped: List[str] = []
        for r in rules_candidate:
            if r in seen:
                continue
            seen.add(r)
            deduped.append(r)
        # 分配id并保存
        self.rules[action_name] = self._assign_rule_ids(action_name, deduped)
        return deduped

    def get_rules_all(self) -> None:
        buffer_pos_path = os.path.join(self.fact_dir, "buffer_correct_temp.json")
        buffer_neg_path = os.path.join(self.fact_dir, "buffer_wrong_temp.json")
        buffer_pos = json.load(open(buffer_pos_path, "r", encoding="utf-8")) if os.path.exists(buffer_pos_path) else {}
        buffer_neg = json.load(open(buffer_neg_path, "r", encoding="utf-8")) if os.path.exists(buffer_neg_path) else {}

        # 加载旧规则
        rules_file = os.path.join(self.rules_dir, "rules_natural_language.json")
        if os.path.exists(rules_file):
            try:
                loaded = json.load(open(rules_file, "r", encoding="utf-8"))
                self.rules = self._normalize_rules_loaded(loaded)
                print(f"[RuleMiner] loaded {sum(len(v) for v in self.rules.values())} existing rules")
            except Exception as e:
                print(f"[RuleMiner] failed to load existing rules: {type(e).__name__}: {e}")
                self.rules = {}

        os.makedirs(self.rules_dir, exist_ok=True)

        # 针对每种错误的action挖掘
        for action, neg_transitions in buffer_neg.items():
            if not isinstance(neg_transitions, list) or not neg_transitions:
                continue
            pos_transitions = buffer_pos.get(action, [])
            merged = list(neg_transitions) + list(pos_transitions or [])
            random.shuffle(merged)
            print(
                f"[RuleMiner] mining action={action} samples={len(merged)} "
                f"(neg={len(neg_transitions)}, pos={len(pos_transitions)})"
            )
            self.get_rules_update(action, merged)

        with open(rules_file, "w", encoding="utf-8") as f:
            json.dump(self.rules, f, indent=2)
