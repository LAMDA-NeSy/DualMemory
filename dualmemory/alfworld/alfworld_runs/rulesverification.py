from utilsextra import *  
from api_config import apply_api_config, get_api_model
import os
import json
import re
import random
import traceback
import time
import hashlib

DEFAULT_RULE_VERIFIER_MODEL = get_api_model("rule_verifier", "deepseek-ai/DeepSeek-V3")


def LLM_request(request, max_retries = 5, initial_retries=5):
    apply_api_config(model_name=DEFAULT_RULE_VERIFIER_MODEL)
    llm = ChatOpenAI(model_name=DEFAULT_RULE_VERIFIER_MODEL, temperature=0) # ! original: gpt-4o, gpt-5-nano, gpt-4o-mini
    if max_retries == 0:
        log_info("************Failed to get workflow. Consider updating your prompt.************\n\n")
        return {}
    try:
        llmrequest_system = "You're an expert in Minecraft. You can help answer some questions related to gameplay mechanics, strategies, and item usage."
        llmrequest_query = request
        messages = [
            SystemMessage(content=llmrequest_system),
            HumanMessage(content=llmrequest_query)
        ]

        llm_response = llm.invoke(messages)
        llm_response = llm_response.content
        llm_response = re.sub(r'^[^\w]+|[^\w]+$', '', llm_response)

        return llm_response
    except Exception as e:
        # Fixed 20 second delay for all retries
        retry_count = initial_retries - max_retries
        delay = 20
        log_info(f"Error encountered: {e}\nWaiting {delay} seconds before retry {retry_count + 1}/{initial_retries}...\n\n")
        
        time.sleep(delay)
        return LLM_request(
            request,
            max_retries=max_retries - 1,
            initial_retries=initial_retries
        )


class RuleVerifier:
    """ Manages state transitions and logs results for analysis. """
    def __init__(self, env_name, io_dir, with_graph=True, model_name=DEFAULT_RULE_VERIFIER_MODEL, temperature=0): # ! original: gpt-4o
        """ Initializes the buffer with a specific model configuration. """

        self.env_name = env_name
        self.io_dir = io_dir
        self.rules_dir = io_dir + '/symbolic_knowledge/'
        self.prompt_dir = io_dir + '/prompts'
        self.fact_dir = io_dir + '/traj_data'
        self.with_graph = with_graph
        apply_api_config(model_name=model_name)
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

        # Load functions from file
        self.functions_set = []
        self.functions_set_string = []
        self.load_functions()


    def deduplicate_rules(self, verbose=True):
        unique_rules = []
        seen_contents = set()

        for rule in self.functions_set_string:
            # Extract the content after "Rule X:" from the natural-language header comment,
            # ignoring numbering/function name differences.
            nl_match = re.search(r"^#\s*Rule\s*\d+\s*:\s*(.+)$", rule, flags=re.MULTILINE)
            if nl_match:
                content_after_colon = nl_match.group(1).strip()
            elif ":" in rule:
                content_after_colon = rule.split(":", 1)[1].strip()
            else:
                continue  # skip strings that do not match the expected format

            # if the content has not been seen, keep the rule
            if content_after_colon not in seen_contents:
                seen_contents.add(content_after_colon)

                unique_rules.append(rule)

        if verbose:
            print(f"[deduplication completed] original number: {len(self.functions_set_string)}, deduplicated number: {len(unique_rules)}")

        self.functions_set_string = unique_rules


    def _sanitize_identifier(self, text: str) -> str:
        text = str(text).strip()
        text = re.sub(r"\W+", "_", text)
        text = text.strip("_")
        if not text:
            return "Rule"
        if not re.match(r"^[A-Za-z_]", text):
            text = f"Rule_{text}"
        return text



    def load_functions(self):
        # Reset loaded functions on every reload to avoid unbounded growth across intervals.
        self.functions_set = []
        self.functions_set_string = []

        code_file = os.path.join(self.rules_dir, self.env_name, 'rules_code.json')
        previous_pruned_code_file = os.path.join(self.rules_dir, self.env_name, 'pruned_rules_code.json')
        if not os.path.exists(code_file):
            print(f"No functions found in {code_file}")
        else:
            with open(code_file, 'r') as f:
                self.functions_set_string = json.load(f)
            if os.path.exists(previous_pruned_code_file):
                with open(previous_pruned_code_file, 'r') as f:
                    previous_pruned_functions_set_string = json.load(f)
                    self.functions_set_string += previous_pruned_functions_set_string
            else:
                print(f"No previous pruned code found in {previous_pruned_code_file}, start from scratch")
            self.deduplicate_rules()
            print(f"Loaded {len(self.functions_set_string)} functions.")
            skipped_count = 0
            for func_str in self.functions_set_string:
                # 规范化：如果没有 rule_id 头部，自动添加
                if not func_str.strip().startswith('# rule_id='):
                    # 从函数名提取 rule_id
                    func_name_match = re.search(r'def\s+(\w+)\s*\(', func_str)
                    if func_name_match:
                        rule_id = func_name_match.group(1)
                        func_str = f"# rule_id={rule_id}\n" + func_str
                
                # Preprocess function string: replace spaces with underscores in function name
                func_def_match = re.search(r'def\s+([\w\s]+)\s*\(', func_str)
                if func_def_match:
                    original_func_name = func_def_match.group(1)
                    # Replace spaces with underscores in function name
                    new_func_name = original_func_name.replace(' ', '_')
                    # Replace the function name in the string
                    func_str = func_str.replace(f'def {original_func_name}(', f'def {new_func_name}(')
                
                # 语法检查：在 exec() 之前先用 compile() 验证代码是否合法
                try:
                    compile(func_str, '<string>', 'exec')
                except SyntaxError as e:
                    # 提取规则名称用于日志
                    rule_name_match = re.search(r'def\s+(\w+)\s*\(', func_str)
                    rule_name = rule_name_match.group(1) if rule_name_match else 'unknown'
                    print(f"[WARNING] Skipping rule '{rule_name}' due to syntax error: {e}")
                    skipped_count += 1
                    continue
                
                # Execute function string to make it available in globals
                exec(func_str, globals())
                # Extract function name using regex (now without spaces)
                func_name = re.search(r'def\s+(\w+)\s*\(', func_str).group(1)
                self.functions_set.append(globals()[func_name])
            
            if skipped_count > 0:
                print(f"[WARNING] Skipped {skipped_count} rules due to syntax errors.")


    def run_all_functions(self, state, action):
        # run all functions in self.functions_set
        for func in self.functions_set:
            feedback, success, suggestion = func(state=state, action=action)
            if not success:  # if the function returns False
                action_result = {
                    "feedback": feedback,
                    "success": success,
                    "suggestion": suggestion
                }
                return action_result
        action_result = {
            "feedback": "You completed the action successfully.",
            "success": True,
            "suggestion": ""
        }
        return action_result


    # record the list of rules that are correctly predicted and the list of rules that are incorrectly predicted
    def functions_verification(self):
        record = {} # key: ruleID; value: list
        record['whole_set'] = []
        # ! verifing with buffer_correct_all.json, only find the rules that are failed
        # 第一阶段：验证【buffer_correct】数据集（执行成功的样本）
        # 目标：确保规则不会误杀真正成功的动作（防止 False Positives）
        # 加载所有成功的样本
        # 只要规则在任何一个成功样本上报错，这个规则基本上就废了（太危险，不能用）
        facts_set = load_json_file(os.path.join(self.fact_dir, self.env_name, 'buffer_correct_all.json'))
        for action, transition_set in facts_set.items():
            for index, transition in enumerate(transition_set):
                # transition['state_0']
                # transition['action']
                # transition['action_result']
                # ! 我们不把“执行成功”的样本加入 'whole_set'，因为我们不想去“因为规则而覆盖”它们。
                # 这里的唯一目的是确保规则不会误判它们（即防止 False Positive）。
                # record['whole_set'].append(action + '_' + str(index)) 
                
                if transition['action'] is None or transition['initial_state'] is None:
                    continue
                # 遍历所有生成的规则
                for func in self.functions_set:
                    func_name = func.__name__
                    try:
                        # 运行规则函数，看是否误杀
                        if self.with_graph:
                            feedback, success, suggestion = func(state=transition['initial_state'], action=transition['action'], scene_graph=transition['sg_info'])
                        else:
                            feedback, success, suggestion = func(state=transition['initial_state'], action=transition['action'])
                    except Exception as e:
                        print(f"\n[Error] Function: {func.__name__} | Transition: {transition['action']['name']} | Index: {index}")
                        print(f"Error Type: {type(e).__name__}")
                        print(f"Error Message: {e}")
                        print("Full Traceback:")
                        traceback.print_exc()

                        record[func_name] = record.get(func_name, []) + ['failed']
                        continue
                    # 如果规则误杀，标记为fail
                    if transition['action_result'] == True and success != transition['action_result']:
                        record[func_name] = record.get(func_name, []) + ['failed']
                    # 如果规则没有误杀，标记为00
                    else:
                        record[func_name] = record.get(func_name, []) + [str(00)]

        # 第二阶段：验证【buffer_wrong】数据集（执行失败的样本）
        # 目标：检查规则能否正确识别出这些无效动作（True Negatives）。
        # 衡量规则捕获执行失败的有效性。
        # ! verifing with buffer_wrong_all.json, find the rules that are failedv and can correct the wrong predictions
        facts_set = load_json_file(os.path.join(self.fact_dir, self.env_name, 'buffer_wrong_all.json'))
        for action, transition_set in facts_set.items():
            for index, transition in enumerate(transition_set):
                # transition['state_0']
                # transition['action']
                # transition['action_result']
                # ! 我们把“执行失败”的样本加入 'whole_set'，因为这些才是规则真正需要从所有动作中识别（覆盖）出来的目标。
                record['whole_set'].append(action + '_' + str(index))
                if transition['action'] is None or transition['initial_state'] is None:
                    continue
                for func in self.functions_set:
                    func_name = func.__name__
                    try:
                        if self.with_graph:
                            feedback, success, suggestion = func(state=transition['initial_state'], action=transition['action'], scene_graph=transition['sg_info'])
                        else:
                            feedback, success, suggestion = func(state=transition['initial_state'], action=transition['action'])
                    except Exception as e:
                        # Skip this function if execution fails, continue to next function
                        print(f"Error executing function {func_name} on transition {action}_{index}: {e}")
                        record[func_name] = record.get(func_name, []) + ['failed']
                        continue
                    
                    # the generated rules are accurate, the generated rules only check if the current transition is infeasible
                    # so when the output is true, it cannot be determined if it did not work or passed the rule
                    # only when the output is false can it be determined if it worked
                    # # -> 规则立大功了！它成功拦截了 World Model 漏掉的错误。
                    # -> 记录样本 ID，比如 "take_5"，表示这个规则解决了第5个take样本的问题。
                    if transition['action_result'] == False and success == transition['action_result']:
                        record[func_name] = record.get(func_name, []) + [action + '_' + str(index)]
                    # means it did not work
                    else:
                        record[func_name] = record.get(func_name, []) + [str(00)]

        verification_result_dir = os.path.join(self.rules_dir, self.env_name, 'verification_result.json')
        with open(verification_result_dir, 'w') as f:
            json.dump(record, f, cls=NumpyEncoder, indent=4)
        # return record


    def replace_rule_number(self, text: str, counter: int) -> str:
        # replace the "# Rule X:" part
        text = re.sub(r"# Rule \d+", f"# Rule {counter}", text)
        # replace the "def Rule_X_" part
        text = re.sub(r"def Rule_\d+_", f"def Rule_{counter}_", text)
        return text


    # will not select duplicate rules, if no rule can cover the uncovered facts, stop further filtering rules and output directly
    # 其核心目标是使用贪心算法（Greedy Algorithm）从验证过的规则集中选出一组“最优”规则。
    # 这组规则需要以最精简的数量，覆盖（检测出）最多的错误案例
    def select_rules(self):
        # 加载验证结果
        verification_result_dir = os.path.join(self.rules_dir, self.env_name, 'verification_result.json')
        data = load_json_file(verification_result_dir)
        # 所有需要被覆盖的目标元素集合（错误案例）
        whole_set = set(data['whole_set'])
        # "whole_set": [
        #     "go to_0", ..., "go to_58",
        #     "take_0", ..., "take_6",
        #     "open_0", ..., "open_17",
        #     "put_0", ..., "put_2",
        #     "cool_0", "cool_1"
    # ]
        # 当前尚未被选中规则覆盖的元素集合
        uncovered_elements = whole_set.copy()
        # 用于存放最终选中的规则
        selected_rules = []

        # prepare the rule set, filter out all rules, and exclude rules that contain 'failed'
        all_rules = {
            key: set([item for item in value if item != '0'])
            for key, value in data.items() 
            if key != 'whole_set' and 'failed' not in value
        }        
        # 一个rule长这样
        # {'Rule_1_': ['0', '0', '0', 'take_8', 'take_9']}

        # use greedy algorithm to select rules
        # 贪心算法选择规则
        # 只要还有未覆盖的元素，就继续选择
        while uncovered_elements:
            # find the rule that can cover the most uncovered elements
            # 本轮找到的最佳规则
            best_rule = None
            # 本轮找到的最佳规则覆盖的元素数量
            best_covered = 0
            # 本轮找到的最佳规则覆盖的元素集合
            best_covered_set = set()
            
            for rule, items in all_rules.items():
                # 遍历所有规则，找到覆盖最多未覆盖元素的规则
                covered = items.intersection(uncovered_elements)
                if len(covered) > best_covered:
                    best_rule = rule
                    best_covered = len(covered)
                    best_covered_set = covered
            
            if not best_rule:
                break  # no rule can increase coverage, stop
            
            # update the selected rules and uncovered elements
            selected_rules.append(best_rule)
            uncovered_elements -= best_covered_set

        # sort all rules by the number of covered elements
        # 按照覆盖元素数量排序
        sorted_rules = sorted(all_rules.items(), key=lambda item: len(item[1]), reverse=True)

        rule_summary = {
            'selected_rules': selected_rules, 
                        'sorted_rules':[(rule, list(items)) for rule, items in sorted_rules]
                        }
        # return the result
        with open(os.path.join(self.rules_dir, self.env_name, 'selected_rules.json'), 'w') as f:
            json.dump(rule_summary, f, cls=NumpyEncoder, indent=4)

        counter = 100
        pruned_functions_set_string = []
        # ! these rules are effective, will be directly used for inference stage
        for rule_index in selected_rules:
            for rule in self.functions_set_string:
                if rule_index in rule:
                    rule = self.replace_rule_number(rule, counter)
                    pruned_functions_set_string.append(rule)
                    counter += 1
        with open(os.path.join(self.rules_dir, self.env_name, 'pruned_rules_code.json'), 'w') as f:
            json.dump(pruned_functions_set_string, f, cls=NumpyEncoder, indent=4)

        counter = 100
        sorted_rules_code = []
        # ! these rules may be effective, but because the current transition data is insufficient, the correcting transition numbers = 0
        for rule_index in sorted_rules:
            for rule in self.functions_set_string:
                if rule_index[0] in rule:
                    rule = self.replace_rule_number(rule, counter)
                    sorted_rules_code.append(rule)
                    counter += 1

        # 把所有的生效的规则code保存到rules_code.json中
        with open(os.path.join(self.rules_dir, self.env_name, 'rules_code.json'), 'w') as f:
            json.dump(sorted_rules_code, f, cls=NumpyEncoder, indent=4)

        # ! sorted_rules: all rules are sorted by the number of uncovered elements
        # ! pruned_functions_set_string: the selected rules are pruned from the original functions_set_string
        
        # Clean up failed and unselected rules
        self.cleanup_unverified_rules(selected_rules, verification_result_dir)
        
        return selected_rules, sorted_rules, pruned_functions_set_string, sorted_rules_code


    def cleanup_unverified_rules(self, selected_rules, verification_result_file):
        """从 rules_natural_language.json 和 rules_code.json 中移除失败的规则
        
        这确保了下一个 interval 使用所有验证过的规则（包括冗余的），但移除了完全失败的规则。
        """
        print("\n[CLEANUP] Starting cleanup of unverified rules...")
        
        # 加载验证结果
        verification_data = load_json_file(verification_result_file)
        
        # 识别失败的规则
        failed_rule_names = set()
        # 识别有效的规则（那些能工作且至少覆盖一种情况的规则）
        valid_rule_names = set()
        
        for rule_name, results in verification_data.items():
            if rule_name == 'whole_set':
                continue
            
            if 'failed' in results:
                failed_rule_names.add(rule_name)
            else:
                # Check if the rule is valid (not all '0' and not failed)
                # In verification_result.json, '0' or '00' means it didn't cover that case
                # If a rule has at least one non-'0' entry, it's valid
                is_valid = False
                for item in results:
                    if item != '0' and item != '00':
                        is_valid = True
                        break
                
                if is_valid:
                    valid_rule_names.add(rule_name)
        
        print(f"[CLEANUP] Found {len(valid_rule_names)} valid rules and {len(failed_rule_names)} failed rules.")

        # Load rules_natural_language.json
        rules_nl_file = os.path.join(self.rules_dir, self.env_name, 'rules_natural_language.json')
        rules_nl = load_json_file(rules_nl_file)
        
        # Clean up rules_natural_language.json
        # Keep rules that are valid (even if redundant)
        cleaned_rules_nl = {}
        removed_count = 0
        kept_count = 0
        
        for action, rule_list in rules_nl.items():
            cleaned_rules_nl[action] = []
            for rule_text in rule_list:
                # Normalize rule entry to {id, text} so we can align by id deterministically.
                if isinstance(rule_text, dict):
                    rule_id = rule_text.get("id") or rule_text.get("rule_id")
                    nl_text = rule_text.get("text") or rule_text.get("rule") or rule_text.get("rule_text")
                else:
                    rule_id = None
                    nl_text = rule_text

                if not nl_text:
                    removed_count += 1
                    continue

                if not rule_id:
                    # Backwards-compatible: derive id from "Rule <num>" + action.
                    num_match = re.match(r"\s*Rule\s+(\d+)\s*[:\-]", str(nl_text), flags=re.IGNORECASE)
                    if num_match:
                        rule_id = f"Rule_{num_match.group(1)}_{self._sanitize_identifier(action)}"

                rule_id = self._sanitize_identifier(rule_id) if rule_id else None

                if rule_id and rule_id in valid_rule_names:
                    cleaned_rules_nl[action].append({"id": rule_id, "text": str(nl_text)})
                    kept_count += 1
                else:
                    removed_count += 1
        
        # Save cleaned rules_natural_language.json
        with open(rules_nl_file, 'w') as f:
            json.dump(cleaned_rules_nl, f, indent=4)
        
        print(f"[CLEANUP] rules_natural_language.json: kept {kept_count} rules, removed {removed_count} rules")
        
        # Clean up rules_code.json
        # Only keep code for valid rules
        rules_code_file = os.path.join(self.rules_dir, self.env_name, 'rules_code.json')
        if os.path.exists(rules_code_file):
            cleaned_rules_code = []
            code_removed = 0
            code_kept = 0
            
            for code_str in self.functions_set_string:
                func_name_match = re.search(r'def\s+(\w+)\s*\(', code_str)
                if func_name_match:
                    rule_name = func_name_match.group(1)
                    # Keep if valid
                    if rule_name in valid_rule_names:
                        cleaned_rules_code.append(code_str)
                        code_kept += 1
                    else:
                        code_removed += 1
            
            # Save cleaned rules_code.json
            with open(rules_code_file, 'w') as f:
                json.dump(cleaned_rules_code, f, cls=NumpyEncoder, indent=4)
            
            print(f"[CLEANUP] rules_code.json: kept {code_kept} rule codes, removed {code_removed} rule codes")
        
        print(f"[CLEANUP] Cleanup complete. Next interval will use all {kept_count} verified valid rules.\n")


    def run(self, state, action):
        for func in self.functions_set:
            if not func(state=state, action=action):  
                return False


    def rule_code_gen(self, action, rule_text, rule_id=None, max_retries=5, initial_retries=5):
        # bool = expectedrulecode(state=state_0, action=action)

        if max_retries == 0:
            log_info("************Failed to get workflow. Consider updating your prompt.************\n\n")
            return {}
        try:
            # ! load_text(os.path.join(self.prompt_dir, f"{prompt}.txt"))
            if self.with_graph:
                prompt_file = os.path.join(self.prompt_dir, 'rule_code_gen_system_with_graph_' + self.env_name + '.txt')
            else:
                prompt_file = os.path.join(self.prompt_dir, 'rule_code_gen_system_' + self.env_name + '.txt')
            rule_code_gen_system = load_text(prompt_file)
            prompt_file = os.path.join(self.prompt_dir, 'rule_code_gen_query.txt')
            rule_code_gen_query = load_text(prompt_file).format(rule = rule_text)
            # rule_code_gen_system = load_prompt("rule_code_gen_system_CC_" + self.env_name) # .replace("<rules>", rules_string)
            # rule_code_gen_query = load_prompt("rule_code_gen_query").format(rule = rule)
            messages = [
                SystemMessage(content=rule_code_gen_system),
                HumanMessage(content=rule_code_gen_query)
            ]

            llm_response = self.llm.invoke(messages)
            response_code = llm_response.content
            # write_task_to_csv( e_metadata)
            # # ensure the format
            # prediction_json = fix_and_parse_json(response_code)
            sanitized_action = self._sanitize_identifier(action)
            if rule_id:
                code_name = self._sanitize_identifier(rule_id)
            else:
                # Backwards-compatible: derive a deterministic id from "Rule <num>" + action.
                num_match = re.match(r"\s*Rule\s+(\d+)\s*[:\-]", str(rule_text), flags=re.IGNORECASE)
                if num_match:
                    code_name = f"Rule_{num_match.group(1)}_{sanitized_action}"
                else:
                    digest = hashlib.md5(str(rule_text).encode("utf-8")).hexdigest()[:8]
                    code_name = f"Rule_{digest}_{sanitized_action}"
            code_str = response_code.replace("python\n", "", 1)
            code_str = code_str.replace("expected_rule_code", code_name)
            code_str = code_str.strip('```')
            code_str = f"# rule_id={code_name}\n# {rule_text}\n" + code_str
            return code_str
            # debug #######################
        except Exception as e:
            # Fixed 20 second delay for all retries
            retry_count = initial_retries - max_retries
            delay = 20
            log_info(f"Error encountered: {e}\nWaiting {delay} seconds before retry {retry_count + 1}/{initial_retries}...\n\n")
            
            time.sleep(delay)
            return self.rule_code_gen(
                action, 
                rule_text,
                rule_id=rule_id,
                max_retries=max_retries - 1,
                initial_retries=initial_retries
            )
    
    def rules_code_all(self):
        # ensure output directory exists
        # rule_set 是一个字典，格式大概是：
        # {"take": [{"id": "...", "text": "Rule 1: ..."}, ...], "open": [...]}
        rule_set = load_json_file(os.path.join(self.rules_dir, self.env_name, 'rules_natural_language.json'))
        generated_rules_code = []
        for action, rules in rule_set.items():
            for rule in rules:
                if isinstance(rule, dict):
                    rule_id = rule.get("id") or rule.get("rule_id")
                    rule_text = rule.get("text") or rule.get("rule") or rule.get("rule_text")
                else:
                    rule_id = None
                    rule_text = rule
                if not rule_text:
                    continue
                rule_code = self.rule_code_gen(action, rule_text, rule_id=rule_id)
                print(f'{rule_code}')
                generated_rules_code.append(rule_code)
        with open(os.path.join(self.rules_dir, self.env_name, 'rules_code.json'), 'w') as f:
            json.dump(generated_rules_code, f, cls=NumpyEncoder, indent=4)
        self.load_functions()


if __name__ == "__main__":

    
    # Initialize the ruleverifier
    env_name = 'alfworld'
    io_dir = os.environ.get("DUALMEMORY_ALFWORLD_DIR") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ruleverifier = RuleVerifier(env_name = env_name, io_dir = io_dir)


    # [Stage 1] rules code generation
    #######################
    ruleverifier.rules_code_all()

    # [Stage 2] rules code verification
    #######################
    ruleverifier.functions_verification()

    # [Stage 3] rules selection
    # selected_rules, sorted_rules, pruned_functions_set_string = ruleverifier.select_rules()
    ruleverifier.select_rules()
