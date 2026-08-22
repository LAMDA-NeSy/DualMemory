import hashlib
import json
import os
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

from api_config import get_api_model
from json_utils import fix_and_parse_json
from utils import get_openai_client


DEFAULT_RULE_VERIFIER_MODEL = get_api_model("rule_verifier", "gpt-4o-mini")


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class RuleVerifier:
    def __init__(
        self,
        env_name: str,
        io_dir: str,
        model_name: str = DEFAULT_RULE_VERIFIER_MODEL,
        temperature: float = 0.0,
    ) -> None:
        self.env_name = env_name
        self.io_dir = io_dir
        self.prompt_dir = os.path.join(io_dir, "prompts")
        self.fact_dir = os.path.join(io_dir, "traj_data", env_name)
        self.rules_dir = os.path.join(io_dir, "symbolic_knowledge", env_name)
        self.model_name = model_name
        self.temperature = float(temperature)

        self.functions_set: List[Any] = []
        self.functions_set_string: List[str] = []
        self.load_functions()

    def _sanitize_identifier(self, text: str) -> str:
        text = str(text).strip()
        text = re.sub(r"\\W+", "_", text).strip("_")
        if not text:
            return "Rule"
        if not re.match(r"^[A-Za-z_]", text):
            text = f"Rule_{text}"
        return text

    def deduplicate_rules(self) -> None:
        unique_rules = []
        seen_contents = set()
        for rule in self.functions_set_string:
            nl_match = re.search(r"^#\\s*Rule\\s*\\d+\\s*:\\s*(.+)$", rule, flags=re.MULTILINE)
            if nl_match:
                content = nl_match.group(1).strip()
            else:
                # Fallback: use the first comment line after rule_id
                lines = [l.strip() for l in rule.splitlines() if l.strip().startswith("#")]
                content = lines[1].lstrip("#").strip() if len(lines) >= 2 else rule
            if content in seen_contents:
                continue
            seen_contents.add(content)
            unique_rules.append(rule)
        self.functions_set_string = unique_rules

    def load_functions(self) -> None:
        self.functions_set = []
        self.functions_set_string = []

        code_file = os.path.join(self.rules_dir, "rules_code.json")
        previous_pruned_code_file = os.path.join(self.rules_dir, "pruned_rules_code.json")

        if os.path.exists(code_file):
            with open(code_file, "r", encoding="utf-8") as f:
                self.functions_set_string = json.load(f) or []
        if os.path.exists(previous_pruned_code_file):
            with open(previous_pruned_code_file, "r", encoding="utf-8") as f:
                self.functions_set_string += (json.load(f) or [])

        if not self.functions_set_string:
            return

        self.deduplicate_rules()

        skipped = 0
        for func_str in list(self.functions_set_string):
            try:
                exec(func_str, globals())
            except SyntaxError:
                skipped += 1
                continue
            except Exception:
                skipped += 1
                continue

            m = re.search(r"def\s+(\w+)\s*\(", func_str)
            if not m:
                continue
            name = m.group(1)
            if name in globals():
                self.functions_set.append(globals()[name])

        if skipped:
            print(f"[RuleVerifier] skipped {skipped} rules due to exec errors")

    def _llm_code(self, system: str, prompt: str) -> str:
        client = get_openai_client(self.model_name)
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=1024,
        )
        return (resp.choices[0].message.content or "").strip()

    def rule_code_gen(self, action: str, rule_text: str, rule_id: Optional[str] = None) -> str:
        system_path = os.path.join(self.prompt_dir, f"rule_code_gen_system_{self.env_name}.txt")
        query_path = os.path.join(self.prompt_dir, "rule_code_gen_query.txt")
        system = _load_text(system_path)
        query = _load_text(query_path).format(rule=rule_text)

        response_code = self._llm_code(system=system, prompt=query)

        sanitized_action = self._sanitize_identifier(action)
        if rule_id:
            code_name = self._sanitize_identifier(rule_id)
        else:
            num_match = re.match(r"\\s*Rule\\s+(\\d+)\\s*[:\\-]", str(rule_text), flags=re.IGNORECASE)
            if num_match:
                code_name = f"Rule_{num_match.group(1)}_{sanitized_action}"
            else:
                digest = hashlib.md5(str(rule_text).encode("utf-8")).hexdigest()[:8]
                code_name = f"Rule_{digest}_{sanitized_action}"

        code_str = response_code
        code_str = code_str.replace("```python", "").replace("```", "").strip()
        code_str = code_str.replace("expected_rule_code", code_name)
        code_str = f"# rule_id={code_name}\n# {rule_text}\n" + code_str.strip() + "\n"
        return code_str

    def rules_code_all(self) -> None:
        # 加载language规则
        rules_file = os.path.join(self.rules_dir, "rules_natural_language.json")
        if not os.path.exists(rules_file):
            raise FileNotFoundError(f"missing rules file: {rules_file}")
        with open(rules_file, "r", encoding="utf-8") as f:
            rule_set = json.load(f) or {}

        generated: List[str] = []
        # 遍历所有language规则
        for action, rules in rule_set.items():
            if not isinstance(rules, list):
                continue
            for rule in rules:
                if isinstance(rule, dict):
                    rid = rule.get("id") or rule.get("rule_id")
                    rtext = rule.get("text") or rule.get("rule") or rule.get("rule_text")
                else:
                    rid = None
                    rtext = str(rule)
                if not rtext:
                    continue
                generated.append(self.rule_code_gen(action, rtext, rule_id=rid))

        os.makedirs(self.rules_dir, exist_ok=True)
        with open(os.path.join(self.rules_dir, "rules_code.json"), "w", encoding="utf-8") as f:
            json.dump(generated, f, indent=2)
        self.load_functions()

    def functions_verification(self) -> None:
        record: Dict[str, Any] = {"whole_set": []}

        correct_path = os.path.join(self.fact_dir, "buffer_correct_all.json")
        wrong_path = os.path.join(self.fact_dir, "buffer_wrong_all.json")
        correct_set = json.load(open(correct_path, "r", encoding="utf-8")) if os.path.exists(correct_path) else {}
        wrong_set = json.load(open(wrong_path, "r", encoding="utf-8")) if os.path.exists(wrong_path) else {}

        # Phase 1: ensure no false positives on correct samples.
        # 遍历正样本集
        for action, transitions in correct_set.items():
            for idx, transition in enumerate(transitions or []):
                state = transition.get("initial_state")
                act = transition.get("action")
                sg = transition.get("sg_info") or {}
                if not state or not act:
                    continue
                # 遍历每条候选规则函数
                for func in self.functions_set:
                    name = func.__name__
                    try:
                        import inspect

                        kwargs = {"state": state, "action": act}
                        sig = None
                        try:
                            sig = inspect.signature(func)
                        except Exception:
                            sig = None
                        if sig is not None and "scene_graph" in sig.parameters:
                            kwargs["scene_graph"] = sg
                        _, success, _ = func(**kwargs)
                    except Exception as e:
                        print(f"[RuleExecError] {name} on correct {action}_{idx}: {e}")
                        traceback.print_exc()
                        record[name] = record.get(name, []) + ["failed"]
                        continue
                    # 误杀正例
                    if transition.get("action_result") is True and bool(success) is False:
                        record[name] = record.get(name, []) + ["failed"]
                    else:
                        record[name] = record.get(name, []) + ["0"]

        # Phase 2: coverage on wrong samples.
        for action, transitions in wrong_set.items():
            for idx, transition in enumerate(transitions or []):
                sample_id = f"{action}_{idx}"
                record["whole_set"].append(sample_id)
                state = transition.get("initial_state")
                act = transition.get("action")
                sg = transition.get("sg_info") or {}
                if not state or not act:
                    continue
                for func in self.functions_set:
                    name = func.__name__
                    try:
                        import inspect

                        kwargs = {"state": state, "action": act}
                        sig = None
                        try:
                            sig = inspect.signature(func)
                        except Exception:
                            sig = None
                        if sig is not None and "scene_graph" in sig.parameters:
                            kwargs["scene_graph"] = sg
                        _, success, _ = func(**kwargs)
                    except Exception as e:
                        print(f"[RuleExecError] {name} on wrong {action}_{idx}: {e}")
                        record[name] = record.get(name, []) + ["failed"]
                        continue
                    # 覆盖负例
                    if transition.get("action_result") is False and bool(success) is False:
                        record[name] = record.get(name, []) + [sample_id]
                    # 没查出来
                    else:
                        record[name] = record.get(name, []) + ["0"]

        out_path = os.path.join(self.rules_dir, "verification_result.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    def replace_rule_number(self, text: str, counter: int) -> str:
        text = re.sub(r"# Rule \\d+", f"# Rule {counter}", text)
        text = re.sub(r"def Rule_\\d+_", f"def Rule_{counter}_", text)
        return text

    def select_rules(self) -> Tuple[List[str], List[Tuple[str, List[str]]], List[str]]:
        # 加载验证结果
        verification_path = os.path.join(self.rules_dir, "verification_result.json")
        with open(verification_path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}

        whole_set = set(data.get("whole_set") or [])
        uncovered = set(whole_set)
        selected: List[str] = []

        all_rules: Dict[str, set] = {
            key: set([item for item in (value or []) if item != "0"])
            for key, value in data.items()
            if key != "whole_set" and isinstance(value, list) and "failed" not in value
        }

        while uncovered:
            best_rule = None
            best_covered_set = set()
            for rule, items in all_rules.items():
                covered = items.intersection(uncovered)
                if len(covered) > len(best_covered_set):
                    best_rule = rule
                    best_covered_set = covered
            if not best_rule:
                break
            selected.append(best_rule)
            uncovered -= best_covered_set

        sorted_rules = sorted(all_rules.items(), key=lambda kv: len(kv[1]), reverse=True)
        rule_summary = {
            "selected_rules": selected,
            "sorted_rules": [(rule, list(items)) for rule, items in sorted_rules],
        }
        with open(os.path.join(self.rules_dir, "selected_rules.json"), "w", encoding="utf-8") as f:
            json.dump(rule_summary, f, indent=2)

        counter = 100
        pruned: List[str] = []
        for rule_name in selected:
            for code in self.functions_set_string:
                if f"def {rule_name}" in code or f"rule_id={rule_name}" in code:
                    pruned.append(self.replace_rule_number(code, counter))
                    counter += 1
                    break

        with open(os.path.join(self.rules_dir, "pruned_rules_code.json"), "w", encoding="utf-8") as f:
            json.dump(pruned, f, indent=2)

        # Clean up failed rules from NL + code so next interval starts from verified ones.
        self.cleanup_unverified_rules(selected_rules=selected, verification_result_file=verification_path)

        return selected, sorted_rules, pruned

    def cleanup_unverified_rules(self, selected_rules: List[str], verification_result_file: str) -> None:
        verification = json.load(open(verification_result_file, "r", encoding="utf-8"))
        failed = set()
        valid = set()
        for rule_name, results in verification.items():
            if rule_name == "whole_set":
                continue
            if not isinstance(results, list):
                continue
            if "failed" in results:
                failed.add(rule_name)
                continue
            if any(item not in {"0", "00"} for item in results):
                valid.add(rule_name)

        rules_nl_file = os.path.join(self.rules_dir, "rules_natural_language.json")
        if os.path.exists(rules_nl_file):
            rules_nl = json.load(open(rules_nl_file, "r", encoding="utf-8")) or {}
            cleaned_nl: Dict[str, List[Dict[str, str]]] = {}
            for action, rule_list in rules_nl.items():
                cleaned_nl[action] = []
                for rule in rule_list or []:
                    if isinstance(rule, dict):
                        rid = rule.get("id") or rule.get("rule_id")
                        rtext = rule.get("text") or rule.get("rule") or rule.get("rule_text")
                    else:
                        rid = None
                        rtext = str(rule)
                    if not rtext:
                        continue
                    rid = self._sanitize_identifier(rid) if rid else None
                    if rid and rid in valid:
                        cleaned_nl[action].append({"id": rid, "text": str(rtext)})
            with open(rules_nl_file, "w", encoding="utf-8") as f:
                json.dump(cleaned_nl, f, indent=2)

        rules_code_file = os.path.join(self.rules_dir, "rules_code.json")
        if os.path.exists(rules_code_file):
            cleaned_code: List[str] = []
            for code_str in self.functions_set_string:
                m = re.search(r"def\s+(\w+)\s*\(", code_str)
                if not m:
                    continue
                rule_name = m.group(1)
                if rule_name in valid:
                    cleaned_code.append(code_str)
            with open(rules_code_file, "w", encoding="utf-8") as f:
                json.dump(cleaned_code, f, indent=2)
