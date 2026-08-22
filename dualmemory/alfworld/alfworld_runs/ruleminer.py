from utilsextra import *  # Import necessary functions and classes from utils module
from api_config import apply_api_config, get_api_model
import os
import json
import random
import datetime
import hashlib
import re
import tiktoken  # Import tiktoken library

DEFAULT_RULE_MINER_MODEL = get_api_model("rule_miner", "deepseek-ai/DeepSeek-V3.2")


class RuleMiner:
    def __init__(self, io_dir, env_name, model_name=DEFAULT_RULE_MINER_MODEL, temperature = 0, choice_num = 1):
        """Initializes the RuleMiner instance with a specific model configuration.
        
        Args:
            model_name (str): The name of the model to use for rule mining.
            temperature (int): The randomness of the model's responses.
        """
        apply_api_config(model_name=model_name)
        self.llm = ChatOpenAI(
            model_name=model_name, 
            temperature=temperature,
            response_format = { "type": "json_object" },
            n = choice_num
            )  # Initialize the LLM model. max_tokens=1024, TODO
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Initialize the tokenizer
        
        self.rules = {}
        self.env_name = env_name
        self.io_dir = io_dir
        self.rules_dir = io_dir + '/symbolic_knowledge/'
        self.prompt_dir = io_dir + '/prompts'
        self.fact_dir = io_dir + '/traj_data'

    def _sanitize_identifier(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\W+", "_", text)
        return text.strip("_")

    def _extract_rule_number(self, rule_text: str) -> str | None:
        # Expected patterns like:
        # "Rule 12: ..." or "Rule 12 - ..."
        match = re.match(r"\s*Rule\s+(\d+)\s*[:\-]", rule_text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _normalize_rules_loaded(self, loaded: dict) -> dict:
        """Normalize rules file content into: {action: [{"id": str, "text": str}, ...]}.
        
        If rule already has an ID, preserve it. Otherwise, generate new ID with content hash.
        """
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
                    if not rule_text:
                        continue
                    if not rule_id:
                        # Generate new ID with content hash for uniqueness
                        content_hash = hashlib.md5(str(rule_text).encode('utf-8')).hexdigest()[:6]
                        rule_number = self._extract_rule_number(str(rule_text))
                        if rule_number:
                            rule_id = f"Rule_{rule_number}_{sanitized_action}_{content_hash}"
                        else:
                            rule_id = f"Rule_{sanitized_action}_{content_hash}"
                    rule_id = self._sanitize_identifier(str(rule_id))
                else:
                    rule_text = str(entry)
                    content_hash = hashlib.md5(rule_text.encode('utf-8')).hexdigest()[:6]
                    rule_number = self._extract_rule_number(rule_text)
                    if rule_number:
                        rule_id = f"Rule_{rule_number}_{sanitized_action}_{content_hash}"
                    else:
                        rule_id = f"Rule_{sanitized_action}_{content_hash}"
                    rule_id = self._sanitize_identifier(str(rule_id))

                # Ensure uniqueness within an action.
                if rule_id in used_ids:
                    suffix = hashlib.md5(str(rule_text).encode("utf-8")).hexdigest()[:8]
                    rule_id = f"{rule_id}_{suffix}"
                used_ids.add(rule_id)

                normalized[action].append({"id": rule_id, "text": rule_text})

        return normalized

    def _assign_rule_ids(self, action: str, rule_texts: list[str]) -> list[dict]:
        """Assign deterministic ids for a list of rule texts for a given action."""
        sanitized_action = self._sanitize_identifier(str(action))
        used_ids: set[str] = set()
        result: list[dict] = []

        for rule_text in rule_texts:
            # Always include content hash for uniqueness
            content_hash = hashlib.md5(rule_text.encode('utf-8')).hexdigest()[:6]
            rule_number = self._extract_rule_number(rule_text)
            
            # Format: Rule_{number}_{action}_{hash} or Rule_{action}_{hash}
            if rule_number:
                rule_id = f"Rule_{rule_number}_{sanitized_action}_{content_hash}"
            else:
                rule_id = f"Rule_{sanitized_action}_{content_hash}"
            
            # Uniqueness should be guaranteed by content hash, but double-check
            if rule_id in used_ids:
                # This should rarely happen, but add full hash as fallback
                full_hash = hashlib.md5(rule_text.encode("utf-8")).hexdigest()[:12]
                rule_id = f"{rule_id}_{full_hash}"
            
            used_ids.add(rule_id)
            result.append({"id": rule_id, "text": rule_text})

        return result

    def _count_tokens(self, text):
        """Counts the number of tokens in a given text.
        
        Args:
            text (str): The text to count tokens for.
        
        Returns:
            int: The number of tokens.
        """
        if isinstance(text, dict):
            text = json.dumps(text)
        return len(self.tokenizer.encode(text))

    def _truncate_tj_buffer(self, tj_buffer, max_tokens):
        """Truncates the tj_buffer to ensure its token count does not exceed the max_tokens limit.
        
        Args:
            tj_buffer (list): The buffer containing transition data.
            max_tokens (int): The maximum allowed tokens.
        
        Returns:
            list: The truncated buffer.
        """
        truncated_buffer = []
        current_tokens = 0
        
        for item in tj_buffer:
            item_tokens = self._count_tokens(item)
            if current_tokens + item_tokens > max_tokens:
                break
            truncated_buffer.append(item)
            current_tokens += item_tokens
        
        return truncated_buffer

    def _extract_between_brackets(self, s):
        # find the index of the first '['
        start_index = s.find('[')
        # find the index of the last ']'
        end_index = s.rfind(']')
        
        # check if '[' and ']' are found, and '[' should be before ']'
        if start_index != -1 and end_index != -1 and start_index < end_index:
            # extract and return the substring
            return s[start_index:end_index + 1]
        else:
            # if no matching '[...]' is found, return an empty string or an appropriate value
            return ""

    def _extract_between_curly_brackets(self, s):
        # find the index of the first '['
        start_index = s.find('{')
        # find the index of the last ']'
        end_index = s.rfind('}')
        
        # check if '[' and ']' are found, and '[' should be before ']'
        if start_index != -1 and end_index != -1 and start_index < end_index:
            # extract and return the substring
            return s[start_index:end_index + 1]
        else:
            # if no matching '[...]' is found, return an empty string or an appropriate value
            return ""

    def _write_to_json(self, file_path, new_data):
        """Append new data to the existing JSON file, handling potential I/O errors."""
        try:
            # Check if file exists and has content; if not, initialize with an empty list
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            except (IOError, json.JSONDecodeError):
                data = []

            data.append(new_data)  # Append new data to the existing data

            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
        except IOError as e:
            print(f"An error occurred while writing to the file: {e}")


    def get_rules_update(self, act_name, tj_buffer, tj_negative, max_retries=5):
        """Attempts to mine rules using the LLM, retrying on failure up to max_retries times.

        Args:
            max_retries (int): Maximum number of retry attempts.

        Returns:
            dict: Parsed rules if successful, or an empty dictionary on failure.
        """

        if max_retries == 0:
            log_info("Failed to get rules after maximum retries. Consider updating your prompt.")
            return {}
        try:
            # 分批处理
            # 因为 Prompt 长度有限，如果样本太多 (total_elements)，不能一次发给 LLM
            total_elements = len(tj_buffer)
            batch_size = 20 # reduced from 100 to fit context window
            for i in range(0, total_elements, batch_size):
                # 取出一个 Batch 的正负样本
                truncated_tj_batch = tj_buffer[i:i + batch_size]
                
                # 构建prompt
                prompt_file = os.path.join(self.prompt_dir, 'rule_improve_system_' + self.env_name + '.txt')
                rule_miner_system = load_text(prompt_file)
                prompt_file = os.path.join(self.prompt_dir, 'rule_improve_query.txt')
                # 这里传入了两个关键信息
                # 1. transitions: 当前这一批样本（包含成功和失败的）。
                # 2. rules: 之前的旧规则 (self.rules.get(act_name, []))，让 LLM 在此基础上改进 
                # 关键：增量更新
                existing_rules = self.rules.get(act_name, [])
                existing_rule_texts = [r["text"] if isinstance(r, dict) and "text" in r else str(r) for r in existing_rules]
                rule_miner_query = load_text(prompt_file).format(transitions=truncated_tj_batch, 
                rules = existing_rule_texts)
                rule_miner_query += f"\n Mining the rules for '{act_name}'"
                messages = [SystemMessage(content=rule_miner_system), HumanMessage(content=rule_miner_query)]


                rules_candidate = []
                # 调用 LLM,注意更改模型
                llm_response = self.llm.generate(messages = [messages])
                # 这里有一个循环是因为 LLM 的 API 支持一次性生成 N 个不同的回答（也就是 Sampling 多次
                # 这个循环是为了收集所有采样结果中的规则
                # 如果 n=1（默认），这个循环只跑 1 次。
                # 如果 n>1（比如为了增加多样性），这个循环会跑 N 次，把所有候选答案里提取出的规则都加到 candidates 里。
                for generation in llm_response.generations[0]:
                    message_content = generation.message.content
                    message_content = self._extract_between_curly_brackets(message_content) 
                    parsed_data = fix_and_parse_json(message_content)
                    rules_temp0 = parsed_data['final_rules']
                    rules_candidate.extend(rules_temp0)
                self.rules[act_name] = self._assign_rule_ids(act_name, rules_candidate)
            return rules_candidate  # Parse the JSON rules and handle errors in formatting.
        except Exception as e:
            log_info(f"Error in Mining Rules: {e}. Retrying...")
            return self.get_rules_update(act_name, tj_buffer, tj_negative, max_retries=max_retries - 1)  # Recursive retry on exception.


    def get_rules_all(self):
        """Attempts to mine rules using the LLM, retrying on failure up to max_retries times.

        Args:
            max_retries (int): Maximum number of retry attempts.

        Returns:
            dict: Parsed rules if successful, or an empty dictionary on failure.
        """
        # 加载buffer数据 temp(这个interval的)
        buffer_pos = load_json_file(os.path.join(self.fact_dir, self.env_name, 'buffer_correct_temp.json'))
        buffer_neg = load_json_file(os.path.join(self.fact_dir, self.env_name, 'buffer_wrong_temp.json'))
        
        # 从文件加载现有规则（为了与 RuleVerifier 的清理保持同步）
        rules_file = os.path.join(self.rules_dir, self.env_name, 'rules_natural_language.json')
        if os.path.exists(rules_file):
            try:
                loaded = load_json_file(rules_file)
                self.rules = self._normalize_rules_loaded(loaded)
                print(f"Loaded {sum(len(v) for v in self.rules.values())} existing rules from file.")
            except Exception as e:
                print(f"Error loading existing rules: {e}")
                self.rules = {}
        
        # 逻辑：只针对有“负样本”（buffer_neg）的动作类型进行挖掘
        for key, value_neg in buffer_neg.items():
            print(f"start mining transactions from {key}")
            # 【关键】将“成功样本”和“失败样本”合并
            # LLM 需要同时看到正反例（Contrastive Learning），才能总结出区别。
            # 比如：为什么有时候 open 冰箱成功了，有时候失败了？
            value_pos = buffer_pos.get(key, [])
            merged_list = value_neg + value_pos
            random.shuffle(merged_list) # Shuffle to mix positive and negative samples for contrastive learning
            self.get_rules_update(key, merged_list, value_neg)
        with open(os.path.join(self.rules_dir, self.env_name, 'rules_natural_language.json'), 'w') as f:
            json.dump(self.rules, f, indent=4)  # Save the rules to a file for debugging.
        # self.rules = {}

if __name__ == "__main__":

    model_name = get_api_model("rule_miner", "deepseek-ai/DeepSeek-V3")
    temperature=0
    io_dir = os.environ.get("DUALMEMORY_ALFWORLD_DIR") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_name = 'alfworld'
    miner = RuleMiner(io_dir = io_dir, env_name = env_name, model_name=model_name, temperature=temperature)  # Create a RuleMiner instance

    miner.get_rules_all()
    print()
