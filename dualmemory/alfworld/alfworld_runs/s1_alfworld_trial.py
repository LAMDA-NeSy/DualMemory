"""Adapted from https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb"""

import os
import sys
import json
import yaml
import shutil
import openai
import importlib
import time
import re
from openai import OpenAI
import alfworld
import alfworld.agents.environment
from utils import Model
from alfworld_prompts import build_action_system_prompt
from env_history import EnvironmentHistory
from scene_graph import *
from stateinfo_transform import process_env_history, state_info_transformation, convert_action

from buffer import Buffer
from ruleminer import RuleMiner
from rulesverification import RuleVerifier
from api_config import apply_api_config, get_api_model, resolve_api_for_model
from task_file import init_ordered_alfworld_env

from typing import List, Dict, Any, Tuple, Optional
 

def render_env_history_for_buffer(env_history: EnvironmentHistory) -> str:
    """
    Render EnvironmentHistory for offline buffering/rule mining.

    In online_rules mode we may inject "Action in Imagination ..." and rule feedback
    into env_history to guide resampling. Those lines are useful for the agent, but
    they can pollute offline state/action extraction. Here we strip them from the
    serialized trajectory saved under traj_data/.../buffer_traj/.
    """
    base_query = str(getattr(env_history, "_cur_query", "")).rstrip("\n")
    history = list(getattr(env_history, "_history", []))

    filtered: List[Dict[str, str]] = []
    i = 0
    while i < len(history):
        item = history[i]
        label = item.get("label")
        value = str(item.get("value", ""))

        if label == "action" and value.startswith("Action in Imagination"):
            # Drop this action and its immediate observation (rule feedback / invalid format).
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


def _default_io_dir() -> str:
    env_dir = os.environ.get("DUALMEMORY_ALFWORLD_DIR")
    if env_dir:
        return env_dir
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_config_path() -> str:
    env_path = os.environ.get("ALFWORLD_CONFIG_PATH")
    if env_path:
        return env_path
    return os.path.join(os.path.dirname(__file__), "base_config.yaml")


########## ! model initialization ##########
apply_api_config()
S1_TRAJECTORY_MODEL = get_api_model("s1_trajectory", "deepseek-ai/DeepSeek-V3")
WORLD_MODEL = get_api_model("world_model", S1_TRAJECTORY_MODEL)
RULE_MINER_MODEL = get_api_model("rule_miner", WORLD_MODEL)
RULE_VERIFIER_MODEL = get_api_model("rule_verifier", WORLD_MODEL)
model_name = S1_TRAJECTORY_MODEL # 'gpt-4o-mini' # 'gpt-4o'
env_name = 'alfworld'
io_dir = _default_io_dir()
buffer = Buffer(io_dir = io_dir, env_name = env_name, model_name = WORLD_MODEL)

interval = 5 # 1
########## ! model initialization ##########

_OPENAI_CLIENTS: Dict[Tuple[str, str], OpenAI] = {}


def _get_openai_client(model_name: str) -> OpenAI:
    settings = resolve_api_for_model(model_name=model_name)
    api_key = settings.get("api_key") or ""
    api_base = settings.get("api_base") or ""
    cache_key = (api_key, api_base)
    cached = _OPENAI_CLIENTS.get(cache_key)
    if cached is not None:
        return cached
    client = OpenAI(api_key=api_key or None, base_url=api_base or None)
    _OPENAI_CLIENTS[cache_key] = client
    return client


FOLDER = io_dir + '/prompts'
PROMPT_FILE = 'alfworld_3prompts.json'
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)

def llm(prompt: str, *, system: str, model: Model, max_tokens: int = 100) -> str:
    stop = ["\r\n", "\n", "\r"]
    max_retries = 10
    for attempt in range(max_retries):
        try:
            resp = _get_openai_client(model).chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
                stop=stop,
            )
            text = (resp.choices[0].message.content or "").strip()
            return text.splitlines()[0]
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error: {e}. Retrying in 10 seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(10)
            else:
                print(f"Error: {e}. Max retries reached.")
                raise

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

def transform_put_inon(action):
    """Normalize put action to in/on (align with ReAct)."""
    put_regex_1 = r"put\s+\w+(\s+\w+)*(\s+\d+)?\s+on\s+\w+(\s+\w+)*(\s+\d+)?"
    put_regex_2 = r"put\s+\w+(\s+\w+)*(\s+\d+)?\s+in\s+\w+(\s+\w+)*(\s+\d+)?"

    if action.startswith("put"):
        if re.search(put_regex_1, action):
            action = action.replace(" on ", " in/on ")
        elif re.search(put_regex_2, action):
            action = action.replace(" in ", " in/on ")
    return action

def transform_put_action(action, transform_to_move_syntax=True):
    """Put action grammar correction (align with ReAct)."""
    if transform_to_move_syntax:
        action = action.replace("put", "move")
        action = action.replace("in/on", "to")
    return action

def alfworld_run(
    env,
    fewshot: str,
    to_print=True,
    ob='',
    model: Model = S1_TRAJECTORY_MODEL,
    feasibility_memory_during_collection: bool = False,
    rules_buffer: Optional[Buffer] = None,
    max_rule_retries: int = 5,
) -> Tuple[EnvironmentHistory, bool]:
    system_prompt = build_action_system_prompt(fewshot=fewshot, max_steps=49)
    env_history = EnvironmentHistory("", ob, [], [])
    env_history.reset()
    if to_print:
        print(ob)
        sys.stdout.flush()
    cur_step = 0



    ##################################################
    # construct initial scene graph with initial observation
    # ! [1. data collection] raw trajectory buffer path 
    ##################################################
    sg = SceneGraph(initialization_info = ob)
    sg_history = []
    # sg.display_graph()
    # sg_file_path = os.path.join(io_dir, 'traj_data', env_name, 'inferenceTime_SG', f'traj_{taskID}', f'transition_info_EnvironmentID{taskID}.json')
    # sg.display_graph()
    ##################################################
    # construct initial scene graph with initial observation
    ##################################################


    while cur_step < 49:
        def _sample_action() -> tuple[str, str]:
            action_text = llm(str(env_history) + ">", system=system_prompt, model=model).strip()
            if action_text.startswith(">"):
                action_text = action_text[1:].strip()
            action_text = transform_put_inon(action_text)
            return action_text, transform_put_action(action_text)

        action, action_for_env = _sample_action()

        # Online collection: use current rules to filter actions, starting from empty rules.
        # This mimics s2's "imagination" loop: add rule feedback to history and resample.
        if feasibility_memory_during_collection and rules_buffer is not None and getattr(rules_buffer, "functions_set", []):
            inner_counter = max(1, int(max_rule_retries))
            imagination_attempt = 0
            last_valid_candidate = action
            last_valid_candidate_for_env = action_for_env

            while inner_counter > 0:
                # Think actions do not interact with the environment and do not need rule checks.
                if action.startswith("think:"):
                    action_for_env = ""
                    break

                state_text = process_env_history(str(env_history))
                try:
                    state_json = state_info_transformation(state_text)
                except Exception:
                    state_json = {}
                if state_json is None:
                    state_json = {}

                action_json = convert_action(action)
                if action_json is None:
                    imagination_attempt += 1
                    env_history.add(
                        "action",
                        f"Action in Imagination (attempt {imagination_attempt}): {action}.",
                    )
                    env_history.add(
                        "observation",
                        "Invalid action format. Please output a single valid action.",
                    )
                    inner_counter -= 1
                    action, action_for_env = _sample_action()
                    continue

                last_valid_candidate = action
                last_valid_candidate_for_env = action_for_env

                rule_check = rules_buffer.worldcode_get_prediction(
                    state_json, action_json, sg.graph
                )
                if rule_check.get("success"):
                    break

                imagination_attempt += 1
                env_history.add(
                    "action",
                    f"Action in Imagination (attempt {imagination_attempt}): {action}.",
                )
                observation = (
                    f"{rule_check.get('feedback', '')}. {rule_check.get('suggestion', '')}"
                ).strip()
                env_history.add("observation", observation)
                inner_counter -= 1
                action, action_for_env = _sample_action()

            if inner_counter <= 0:
                # Give up after retries and execute the last sampled valid-format action anyway.
                action = last_valid_candidate or "look"
                action_for_env = last_valid_candidate_for_env

        # List of valid action keywords
        valid_actions = ["go to", "open", "close", "take", "put", "clean", "heat", "cool", "use", "look", "inventory"]

        env_history.add("action", action)
        # Keep original s1 behavior unless online rule-guided collection is enabled.
        if action.startswith("think:") and feasibility_memory_during_collection:
            observation = "OK."
            done = False
        else:
            observation, reward, done, info = env.step([action_for_env])
            observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
            if action.startswith("think:"):
                observation = "OK."
        env_history.add("observation", observation)


        ##################################################
        # update scene graph with action and observation at each step
        # ! [1. data collection] raw trajectory buffer path 
        ##################################################
        for action_keyword in valid_actions:
            if action.startswith(action_keyword):
                sg_history.append(copy.deepcopy(sg.graph))
                interaction_info = '> ' + action + '\n' + observation
                sg.update_graph(interaction_info)
        # sg.display_graph()
        ##################################################
        # update scene graph with action and observation at each step
        ##################################################

    
        if to_print:
            step_idx = cur_step + 1
            print(f'Act {step_idx}: {action}\nObs {step_idx}: {observation}')
            sys.stdout.flush()
        if done:
            return env_history, True, sg_history
        cur_step += 1
    return env_history, False, sg_history

PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

def run_trial(
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        model: Model,
        defer_rules: bool = False,
        online_rules: bool = False,
        task_file: str = "",
    ) -> List[Dict[str, Any]]:
    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)

    miner = None
    ruleverifier = None
    if not defer_rules:
        miner = RuleMiner(io_dir=io_dir, env_name=env_name, model_name=RULE_MINER_MODEL)
        ruleverifier = RuleVerifier(
            env_name=env_name,
            io_dir=io_dir,
            model_name=RULE_VERIFIER_MODEL,
        )


    # In online mode, start from an empty rule set (even if prior runs left rules on disk).
    if online_rules:
        buffer.functions_set = []
        print("[OnlineRules] Enabled: starting from empty rules.")


    ####################### ! clear buffer_traj and buffer_SG #######################
    # Delete all files and subdirectories in buffer_traj after processing all trajectories
    buffer_traj_dir = os.path.join(io_dir, 'traj_data', env_name, 'buffer_traj')
    if os.path.exists(buffer_traj_dir):
        for item in os.listdir(buffer_traj_dir):
            item_path = os.path.join(buffer_traj_dir, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")

    buffer_sg_dir = os.path.join(io_dir, 'traj_data', env_name, 'buffer_SG')
    if os.path.exists(buffer_sg_dir):
        for item in os.listdir(buffer_sg_dir):
            item_path = os.path.join(buffer_sg_dir, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")
    ####################### ! clear buffer_traj and buffer_SG #######################




    with open(_default_config_path()) as reader:
        config = yaml.safe_load(reader)
    split = "train"

    if not task_file:
        raise ValueError(
            "An ordered ALFWorld task file is required for reproducible Stage 1 collection."
        )
    env = init_ordered_alfworld_env(
        config=config,
        environment_type=config["env"]["type"],
        split=split,
        task_file=task_file,
        expected_num_envs=len(env_configs),
    )

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)

    # 遍历task
    for z, env_config in enumerate(env_configs):
        task_id = z
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])

        print(f"using {name}")

        if env_config["is_success"]:
            num_successes += 1

            # log to world log
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            continue

        # 判断任务类型 
        for i, (k, v) in enumerate(PREFIXES.items()):
            if name.startswith(k):
                # Few-shot examples go into the system prompt (align with WebShop s3_main).
                task_examples = d[f"react_{v}_1"] + d[f"react_{v}_0"]
                # 跑一个task
                final_env_history, is_success, sg_history = alfworld_run(
                    env,
                    task_examples,
                    to_print=True,
                    ob=ob,
                    model=model,
                    feasibility_memory_during_collection=online_rules,
                    rules_buffer=buffer,
                )

                # update env config
                if is_success:
                    status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
                    env_configs[z]['is_success'] = True
                    num_successes += 1
                    num_additional_successes += 1
                else:
                    status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'

                # log to world log
                with open(world_log_path, 'a') as f:
                    f.write(status_str + '\n')

                # log env results to trial log
                with open(trial_log_path, 'a') as wf:
                    wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')
                
                print(f'look: i: {i}, k: {k}, v: {v}')


                ####################
                # 这里保存的trajectory是纯文本，虽然写入的json文件。
                # ! [1. data collection] raw trajectory buffer path 
                file_path = os.path.join(io_dir, 'traj_data', env_name, 'buffer_traj', f'traj_{z}', f'transition_info_EnvironmentID{z}_Trial#{trial_idx}.json')
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w') as f:
                    # json.dump(final_env_history, f, cls=NumpyEncoder, indent=4)
                    f.write(render_env_history_for_buffer(final_env_history))
                print(f'look: file_path: {file_path}')

                sg_history_file = os.path.join(io_dir, 'traj_data', env_name, 'buffer_SG', f'traj_{z}', f'sg_transition_info_EnvironmentID{z}_Trial#{trial_idx}.json')
                os.makedirs(os.path.dirname(sg_history_file), exist_ok=True)
                # with open(sg_history_file, 'w') as f:
                #     f.write(str(sg_history))
                with open(sg_history_file, 'w') as f:
                    json.dump(sg_history, f)
                print(f'look: sg_history_file: {sg_history_file}')
                ####################

        # 当处理了interval个task后，进行一次rule mining和rule verification
        if not defer_rules:
            if (task_id + 1) % interval == 0 or (task_id + 1 == num_envs and (task_id + 1) % interval):
                if task_id + 1 == num_envs and (task_id + 1) % interval:
                    current_interval = (task_id + 1) % interval
                else:
                    current_interval = interval
                start_task_id = max(0, task_id + 1 - current_interval)

                ## Stage : transition buffering
                # ! [1. data collection] raw trajectory --> json format
                # 将上述raw text trajectory转换为结构化的(state, action, result) transition
                # 包含 LLM 作为 World Model 预测动作结果（LLM模型要手动更改，目前是DeepSeek-V3.2）
                # Keep raw trajectories for analysis/debugging; do not delete buffer_traj/buffer_SG per-interval.
                buffer.string_buffer_for_transitions_pure(current_interval, start_task_id, cleanup=False)

                # ! [2. rules mining] rules mining
                miner.get_rules_all()

                # [Stage 1] rules code generation
                #######################
                ruleverifier.rules_code_all()
                # [Stage 2] rules code verification
                #######################
                # 规则验证：拿所有的候选规则去跑一边之前收集的样本
                ruleverifier.functions_verification()
                # [Stage 3] rules selection
                # selected_rules, sorted_rules, pruned_functions_set_string = ruleverifier.select_rules()
                ruleverifier.select_rules()

                # Online mode: refresh in-memory rule functions for subsequent episodes.
                if online_rules:
                    try:
                        buffer.functions_set = []
                        if os.path.exists(buffer.rule_code_file):
                            buffer.load_functions_from_file(buffer.rule_code_file)
                        print(f"[OnlineRules] Reloaded rules: {len(buffer.functions_set)} functions.")
                    except Exception as e:
                        print(f"[RuleReloadError] {type(e).__name__}: {e}")

        if task_id >= 100:
            break
        ####################



    # close environment object
    env.close()

    # log trial results to trial and world logs
    log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
-----"""
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return env_configs
