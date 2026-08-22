from utilsextra import *
import os
import json
import shutil
from stateinfo_transform import *

class Buffer:
    """ Manages state transitions and logs results for analysis. """
    def __init__(self, io_dir, env_name, model_name=None, temperature=0):
        self.io_dir = io_dir
        self.env_name = env_name
        self.prompt_dir = io_dir + '/prompts'
        self.trajData_dir = io_dir + '/traj_data'
        self.rules_dir = io_dir + '/symbolic_knowledge/'
        self.record_wrong = {}
        self.record_correct = {}

        self.functions_set = []
        self.rule_code_file = os.path.join(self.rules_dir, self.env_name, 'pruned_rules_code.json')
        if os.path.exists(self.rule_code_file):
            self.load_functions_from_file(self.rule_code_file)


    def load_functions_from_file(self, code_file):
        # load functions from file
        with open(code_file, 'r') as f:
            function_strings = json.load(f)
        
        for func_str in function_strings:
            # convert the string形式的函数 to executable function
            exec(func_str, globals())
            # use regex to extract the function name
            func_name = re.search(r'def\s+(\w+)\s*\(', func_str).group(1)
            self.functions_set.append(globals()[func_name])

    def run_all_functions(self, state, action, sg=None):
        # run all functions in self.functions_set

        for func in self.functions_set:
            # Be robust at inference time: malformed state/action should not crash the whole run.
            if state is None:
                state = {}
            if action is None:
                action = {}
            if sg is None:
                sg = {}
            try:
                feedback, success, suggestion = func(state=state, action=action, scene_graph=sg)
            except Exception as e:
                # Skip rules that error out on this particular input.
                # Rule selection/verification should prevent most of these, but inference-time
                # state parsing can be incomplete.
                print(f"[RuleError] {func.__name__}: {type(e).__name__}: {e}")
                continue
            if not success:  # if the function returns False
                action_result = {
                    "feedback": f"[{func.__name__}] {feedback}".strip(),
                    "success": success,
                    "suggestion": suggestion,
                    "rule_id": func.__name__,
                }
                return action_result
        action_result = {
            "feedback": "You completed the action successfully.",
            "success": True,
            "suggestion": "",
            "rule_id": "",
        }
        return action_result
    

    # ! [3. test with code rules] get world code prediction
    def worldcode_get_prediction(self, state, action, sg):
        feedback = "You completed the action successfully."
        act_success = True
        suggestion = ''
        rule_id = ""

        if self.functions_set:
            success = self.run_all_functions(state, action, sg)
            if not success['success']:
                print("!!!!!!!!!rules code predict fail")
                # if rule violation is detected
                act_success = success['success']
                feedback = success['feedback']
                suggestion = success['suggestion']
                rule_id = success.get("rule_id", "")

        action_result = {
            "feedback": feedback,
            "success": act_success,
            "suggestion": suggestion,
            "rule_id": rule_id,
        }

        return action_result


    def string_buffer_for_transitions_pure(self, interval, task_id, cleanup=True):
        """ Processes transitions from a task directory and logs them. """
        record_correct_temp = {}
        record_wrong_temp = {}
        # 遍历interval次
        for kk in range(interval):
            trajectory_dir = os.path.join(self.trajData_dir, self.env_name, 'buffer_traj', f'traj_{task_id+kk}')
            sg_dir = os.path.join(self.trajData_dir, self.env_name, 'buffer_SG', f'traj_{task_id+kk}')
            files = os.listdir(trajectory_dir)
            # Filter the list to include only JSON files
            # 这里的json文件实际上是之前写入的纯文本日志
            json_files = [f for f in files if f.endswith('.json')]
            for json_file in json_files:
                file_path = os.path.join(trajectory_dir, json_file)
                with open(file_path, 'r', encoding='utf-8') as file:
                    # input_str = json.load(file)
                    input_str = file.read()
                sg_str = load_json_file(os.path.join(sg_dir, 'sg_' + json_file))
                
                # Step 1: Remove everything before "Here is the task:"
                # 清除 Here is the task 之前的内容
                task_start_idx = input_str.find("Here is the task:")
                if task_start_idx == -1:
                    task_start_idx = input_str.find("Here is the task.")
                if task_start_idx != -1:
                    input_str = input_str[task_start_idx:]
                
                # Step 2: Initialize variables
                state_text = ""
                action_text = ""
                action_result = False
                # List of valid action keywords
                valid_actions = ["go to", "open", "close", "take", "put", "clean", "heat", "cool", "use", "look", "inventory"]
                
                # Step 3: Split input string into lines and iterate through them
                lines = input_str.splitlines()
                transition_counter = 0
                for i, line in enumerate(lines):
                    # Step 4: Check if line starts with '>'
                    # 识别动作行：以'>'开头
                    if line.strip().startswith('>'):
                        # Step 5: Strip leading non-alphanumeric characters and check for valid actions
                        # 提取动作
                        stripped_line = line.lstrip("> ").strip()
                        # 确认是合法动作
                        for action_keyword in valid_actions:
                            if stripped_line.startswith(action_keyword):
                                action_key = "put" if action_keyword == "move" else action_keyword
                                # Store the action
                                action_text = stripped_line
                                # Store the state (everything before this line)
                                state_text = "\n".join(lines[:i])
                                # Step 6: Determine the result (check the next line)
                                # Find the position of the first newline character
                                newline_index = state_text.find('\n')
                                # Remove everything before and including the first newline character
                                if newline_index != -1:
                                    state_text = state_text[newline_index + 1:]
                                
                                # 如果是Nothing happens，说明动作失败
                                if i + 1 < len(lines) and lines[i + 1].strip() == "Nothing happens.":
                                    action_result = False
                                else:
                                    action_result = True

                                # state transfer:
                                ##################
                                # 将纯文本的状态和动作转化为结构化表示
                                state = state_info_transformation(state_text)
                                action = convert_action(action_text)

                                # pure collection:
                                ##################
                                # add scene graph information at each time step
                                sg_info = sg_str[transition_counter]
                                transition_info = {'initial_state': state, 'action': action, 'action_result': action_result, 'sg_info': sg_info}
                                transition_counter += 1
                                # record_c_w.setdefault(action_keyword, []).append(transition_info)

                                if not action_result:
                                    # 失败 -> record_wrong
                                    self.record_wrong.setdefault(action_key, []).append(transition_info)
                                    record_wrong_temp.setdefault(action_key, []).append(transition_info)
                                else:
                                    # 成功 -> record_correct
                                    self.record_correct.setdefault(action_key, []).append(transition_info)
                                    record_correct_temp.setdefault(action_key, []).append(transition_info)

                                # with open(os.path.join(self.trajData_dir, self.env_name, 'trajrecord.log'), 'a') as wf:
                                #     wf.write(f"\n[State text]: {state_text}")
                                #     wf.write(f"\n[State json]: {state}")
                                #     wf.write(f"\n[Action]: {action}")
                                #     wf.write(f"\n[Action Result]: {action_result}")
                                #     # wf.write(f"\n[Predicted Action Result]: {predicted_state_1}")
                                #     wf.write("\n--------------------------------------------------------\n")
                                break

            # 写入json文件
            with open(os.path.join(self.trajData_dir, self.env_name, 'buffer_wrong_all.json'), 'w') as f:
                json.dump(self.record_wrong, f, indent=4)
            with open(os.path.join(self.trajData_dir, self.env_name, 'buffer_correct_all.json'), 'w') as f:
                json.dump(self.record_correct, f, indent=4)

            with open(os.path.join(self.trajData_dir, self.env_name, 'buffer_wrong_temp.json'), 'w') as f:
                json.dump(record_wrong_temp, f, indent=4)
            with open(os.path.join(self.trajData_dir, self.env_name, 'buffer_correct_temp.json'), 'w') as f:
                json.dump(record_correct_temp, f, indent=4)

            # with open(os.path.join(self.trajData_dir, self.env_name, 'buffer_fact', f'buffer_c_w_taskID{task_id}_interval{interval}.json'), 'w') as f:
            #     json.dump(record_c_w, f, indent=4)
            # with open(os.path.join(self.trajData_dir, self.env_name, 'buffer_fact', f'buffer_prediction_record_taskID{task_id}_interval{interval}.json'), 'w') as f:
            #     json.dump(record, f, indent=4)
            # with open(os.path.join(self.trajData_dir, self.env_name, 'buffer_fact', f'buffer_wrong_prediction_taskID{task_id}_interval{interval}.json'), 'w') as f:
            #     json.dump(record_wrong_prediction, f, indent=4)
        
        # Delete processed trajectories if requested.
        if cleanup:
            for kk in range(interval):
                trajectory_dir = os.path.join(self.trajData_dir, self.env_name, 'buffer_traj', f'traj_{task_id+kk}')
                sg_dir = os.path.join(self.trajData_dir, self.env_name, 'buffer_SG', f'traj_{task_id+kk}')
                for path in (trajectory_dir, sg_dir):
                    if not os.path.exists(path):
                        continue
                    try:
                        shutil.rmtree(path)
                    except Exception as e:
                        print(f"Error deleting {path}: {e}")


    def update_rules(self, rules_extra):
        self.rules = rules_extra
