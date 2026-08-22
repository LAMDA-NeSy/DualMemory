"""Adapted from https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb"""

import os
import sys
import json
import yaml
import openai
import importlib
import time
import alfworld
import alfworld.agents.environment
import re
from openai import OpenAI
from env_history import EnvironmentHistory
from scene_graph import *
from stateinfo_transform import *
from utils import *
from alfworld_prompts import build_action_system_prompt
from utilsextra import *
from progress_memory import (
    ProgressMemoryPlanner,
    build_action_prompt,
    extract_visited_locations_from_trajectory,
    format_visited_locations,
)

from buffer import Buffer
from api_config import apply_api_config, get_api_model, resolve_api_for_model
from task_file import init_ordered_alfworld_env

from typing import List, Dict, Any, Tuple, Optional
 
def render_env_history_with_progress_memory_events(
    env_history: EnvironmentHistory,
    events: List[tuple[int, str]],
) -> str:
    """Render EnvironmentHistory with extra log events inserted at history indices."""
    history = list(getattr(env_history, "_history", []))
    base_query = str(getattr(env_history, "_cur_query", "")).rstrip("\n")

    bucket: Dict[int, List[str]] = {}
    for idx, msg in events or []:
        bucket.setdefault(int(idx), []).append(str(msg))

    lines: List[str] = []
    if base_query:
        lines.append(base_query)

    for i, item in enumerate(history):
        for msg in bucket.get(i, []):
            lines.append(msg.rstrip("\n"))

        label = item.get("label")
        value = str(item.get("value", ""))
        if label == "action":
            lines.append(f"> {value}")
        elif label == "observation":
            lines.append(value)
        elif label == "human_edit":
            lines.append(f"[human edit]: {value}")

    for msg in bucket.get(len(history), []):
        lines.append(msg.rstrip("\n"))

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
S3_TRAJECTORY_MODEL = get_api_model("s3_trajectory", "deepseek-ai/DeepSeek-V3")
WORLD_MODEL = get_api_model("world_model", S3_TRAJECTORY_MODEL)
model_name = S3_TRAJECTORY_MODEL # 'gpt-4o-mini' # 'gpt-4o'
env_name = 'alfworld'
io_dir = _default_io_dir()
buffer = Buffer(io_dir = io_dir, env_name = env_name, model_name = WORLD_MODEL)
interval = 1 # 5
wm_with_graph = True # False
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

def extract_task_text(ob: str) -> str:
    lines = [ln.strip() for ln in (ob or "").splitlines() if ln.strip()]

    # Prefer the explicit ALFWorld task line (e.g., "Your task is to: ...").
    for line in lines:
        low = line.lower()
        if low.startswith("your task is to") or "your task is to:" in low:
            return line

    # Some variants use "Task: ..."
    for line in lines:
        low = line.lower()
        if low.startswith("task:"):
            return line

    # Fallback: first line mentioning "task", but skip the generic header "Here is the task."
    for line in lines:
        low = line.lower().strip()
        if low in {"here is the task.", "here is the task:", "here is the task"}:
            continue
        if "task" in low:
            return line

    return (ob or "").strip()

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
            if not text:
                raise ValueError("Received empty response from LLM")
            return text.splitlines()[0]  # 兜底强制一行
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 30 if "empty response" in str(e) else 20
                print(
                    f"Error: {e}. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})"
                )
                sys.stdout.flush()
                time.sleep(wait_time)
            else:
                print(f"Error: {e}. Max retries reached.")
                sys.stdout.flush()
                raise

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

def transform_put_inon(action: str) -> str:
    """Normalize put action to in/on (align with ReAct)."""
    put_regex_1 = r"put\s+\w+(\s+\w+)*(\s+\d+)?\s+on\s+\w+(\s+\w+)*(\s+\d+)?"
    put_regex_2 = r"put\s+\w+(\s+\w+)*(\s+\d+)?\s+in\s+\w+(\s+\w+)*(\s+\d+)?"

    if action.startswith("put"):
        if re.search(put_regex_1, action):
            action = action.replace(" on ", " in/on ")
        elif re.search(put_regex_2, action):
            action = action.replace(" in ", " in/on ")
    return action


def transform_put_action(action: str, transform_to_move_syntax: bool = True) -> str:
    """Put action grammar correction (align with ReAct)."""
    if transform_to_move_syntax:
        action = action.replace("put", "move")
        action = action.replace("in/on", "to")
    return action


def normalize_action_for_react(action_orig: str) -> tuple[str, str]:
    """
    Align with s1_alfworld_trial:
    - env_history stores ReAct-style "put ... in/on ..."
    - env.step executes env-style "move ... to ..."
    """
    action_orig = action_orig.strip()

    # Handle move -> (history put, env move)
    if action_orig.startswith("move"):
        action_history = "put" + action_orig[len("move"):]
        action_history = action_history.replace(" to ", " in/on ")
        action_history = transform_put_inon(action_history)
        action_for_env = transform_put_action(action_history, transform_to_move_syntax=True)
        return action_history, action_for_env

    # Handle put -> (history put, env move)
    if action_orig.startswith("put"):
        action_history = transform_put_inon(action_orig)
        action_for_env = transform_put_action(action_history, transform_to_move_syntax=True)
        return action_history, action_for_env

    # Other actions: same for history and env.
    return action_orig, action_orig

## Note: do not add hand-written action gating here.
## Per paper setting, PROGRESS_MEMORY provides LLM-generated global guidance; local validity is handled by the rule system.

def alfworld_run(
    env,
    base_prompt,
    taskID: int,
    to_print=True,
    ob='',
    model: Model = S3_TRAJECTORY_MODEL,
    progress_memory_planner: ProgressMemoryPlanner = None,
    task_examples: str = "",
    trial_log_path: str = "",
    use_local_fewshot: bool = False,
    feasibility_memory: bool = True,
) -> Tuple[EnvironmentHistory, bool, List[Any], List[tuple[int, str]]]:
    system_prompt = build_action_system_prompt(fewshot=task_examples, max_steps=49)
    env_history = EnvironmentHistory(base_prompt, ob, [], [])
    env_history.reset()
    if to_print:
        print(ob)
        sys.stdout.flush()
    cur_step = 0
    trajectory_records = []
    start_observation = ob
    # 任务描述
    task_text = extract_task_text(ob)
    milestone_guide: List[str] = []
    current_milestone_idx = 0
    progress_memory_log_events: List[tuple[int, str]] = []
    logged_local_fewshot_for: set[int] = set()

    # 先用progress_memory将任务拆解成High-level milestones (using llm)
    # 这里修复了，改成了只在当前任务的动作类型里找
    # 【PROGRESS_MEMORY】 全局指导 
    if progress_memory_planner and progress_memory_planner.library.has_data():
        milestone_guide = progress_memory_planner.build_milestone_guide(task_text)
        if milestone_guide:
            log_message = f"\n{'='*60}\n"
            log_message += f"[PROGRESS_MEMORY] Task分解成 {len(milestone_guide)} 个Milestones:\n"
            for idx, milestone in enumerate(milestone_guide, 1):
                log_message += f"  {idx}. {milestone}\n"
            log_message += f"{'='*60}\n"

            print(log_message)
            sys.stdout.flush()
            # idx=0 => insert right after task header, before first action.
            progress_memory_log_events.append((0, log_message))
    # 大致长这样
    # [
    #     "Find and pick up the apple",                 # 第 1 步：先找苹果并拿到手
    #     "Go to the microwave and heat the apple",    # 第 2 步：去微波炉加热
    #     "Go to the fridge",                          # 第 3 步：走向冰箱
    #     "Place the heated apple in the fridge"       # 第 4 步：把热苹果放进去
    # ]


    ##################################################
    # construct initial scene graph with initial observation
    # ! [1. data collection] raw trajectory buffer path 
    # ! [3. inference time] read sg from file
    ##################################################
    sg = SceneGraph(initialization_info = ob)
    sg_history = []
    sg_file_path = os.path.join(io_dir, 'traj_data', env_name, 'inferenceTime_SG', f'traj_{taskID}', f'transition_info_EnvironmentID{taskID}.json')
    sg.load_from_json(sg_file_path) # _Trial#{trial_idx}
    ##################################################
    # construct initial scene graph with initial observation
    ##################################################



    valid_actions = ["go to", "open", "close", "take", "put", "clean", "heat", "cool", "use", "look", "inventory"]

    while cur_step < 50:

        action_success = False
        action_for_env = ""
        
        
        # 【PROGRESS_MEMORY】判断当前milestone（迭代检查直到无法再推进）
        if progress_memory_planner and milestone_guide and trajectory_records:
            prev_milestone_idx = current_milestone_idx
            
            # 迭代检查，直到无法再推进
            max_iterations = len(milestone_guide)  # 最多推进到所有 milestone 完成
            for iteration in range(max_iterations):
                new_milestone_idx = progress_memory_planner.determine_current_milestone(
                    task_text=task_text,
                    trajectory=trajectory_records,
                    milestone_guide=milestone_guide,
                    current_milestone_idx=current_milestone_idx,
                )
                
                # Debug: record the last progress check decision (YES/NO + evidence step index)
                try:
                    check = getattr(progress_memory_planner, "last_progress_check", {}) or {}
                    if check.get("raw"):
                        details = []
                        if check.get("evidence"):
                            details.append(f"evidence={str(check.get('evidence')).strip()}")
                        if check.get("reason"):
                            details.append(f"reason={str(check.get('reason')).strip()}")
                        details_text = f"\n[PROGRESS_MEMORY] Step {cur_step}: Progress details: " + " | ".join(details) + "\n" if details else ""
                        log_message = (
                            f"\n[PROGRESS_MEMORY] Step {cur_step}: Milestone progress check (iter {iteration+1}): {check.get('raw')}\n"
                            f"{details_text}"
                        )
                        progress_memory_log_events.append((len(env_history._history), log_message))
                except Exception:
                    pass
                
                # 如果没有推进，停止迭代
                if new_milestone_idx == current_milestone_idx:
                    break
                
                # 更新 milestone 并记录切换
                old_idx = current_milestone_idx
                current_milestone_idx = new_milestone_idx
                log_message = f"\n[PROGRESS_MEMORY] Step {cur_step}: Milestone切换 (iter {iteration+1}): {old_idx+1} -> {current_milestone_idx+1}\n"
                log_message += f"[PROGRESS_MEMORY] 当前Milestone ({current_milestone_idx+1}/{len(milestone_guide)}): {milestone_guide[current_milestone_idx]}\n"
                
                print(log_message)
                sys.stdout.flush()
                progress_memory_log_events.append((len(env_history._history), log_message))
                
                # 如果已经到达最后一个 milestone，停止迭代
                if current_milestone_idx >= len(milestone_guide) - 1:
                    break


        

        def _env_history_to_steps(history: EnvironmentHistory) -> List[Dict[str, str]]:
            """Convert EnvironmentHistory internal log into (action, observation) steps.

            This includes rule-feedback / imagination attempts added into env_history, so the LLM
            can react to them during retry loops.
            """
            steps: List[Dict[str, str]] = [
                {"action": "(start)", "observation": str(start_observation)}
            ]
            raw = list(getattr(history, "_history", []) or [])
            i = 0
            while i < len(raw) - 1:
                if raw[i].get("label") == "action" and raw[i + 1].get("label") == "observation":
                    steps.append(
                        {
                            "action": str(raw[i].get("value", "")),
                            "observation": str(raw[i + 1].get("value", "")),
                        }
                    )
                    i += 2
                    continue
                i += 1
            return steps

        def _build_action_prompt() -> str:
            if not (progress_memory_planner and milestone_guide):
                return str(env_history).rstrip() + "\n>"

            current_milestone = milestone_guide[current_milestone_idx]
            hint = ""  # step_hint removed (--use_step_hint param deleted)
            milestone_demos = ""

            if use_local_fewshot:
                try:
                    milestone_demos = progress_memory_planner.build_local_fewshot(
                        task_text=task_text,
                        trajectory=trajectory_records,
                        milestone_guide=milestone_guide,
                        milestone_idx=current_milestone_idx,
                        max_demo_steps=0,  # no truncation (debugging/analysis)
                    )
                except Exception:
                    milestone_demos = ""

                if current_milestone_idx not in logged_local_fewshot_for:
                    logged_local_fewshot_for.add(current_milestone_idx)
                    diag = (
                        "\n"
                        + "=" * 60
                        + "\n"
                        + "[PROGRESS_MEMORY][LOCAL_FEWSHOT] Injected milestone-level demonstrations\n"
                        + f"Milestone ({current_milestone_idx + 1}/{len(milestone_guide)}): {current_milestone}\n"
                        + "\n----- BEGIN MILESTONE DEMOS (full; masked IDs) -----\n"
                        + (milestone_demos or "- None")
                        + "\n----- END MILESTONE DEMOS -----\n"
                        + "=" * 60
                        + "\n"
                    )
                    history_idx = len(getattr(env_history, "_history", []))
                    progress_memory_log_events.append((history_idx, diag))

            # Align with WebShop s3_main: use a prompt template (progress_memory_action_prompt.txt) instead of string concatenation.
            return build_action_prompt(
                env_history=str(env_history),
                milestone_guide=milestone_guide,
                hint=hint,
                task_text=task_text,
                trajectory=_env_history_to_steps(env_history),
                milestone_demos=milestone_demos,
                current_milestone=current_milestone,
                prompt_dir=FOLDER,
                max_steps=0,  # include full trajectory text
            )

        action_prompt = _build_action_prompt()

        if not feasibility_memory:
            action_orig = llm(action_prompt, system=system_prompt, model=model).strip()
            action_orig = re.sub(r'^[^a-zA-Z]+', '', action_orig)
            action, action_for_env = normalize_action_for_react(action_orig)

        else:
            inner_counter = 5
            imagination_attempt = 0
            last_valid_candidate = ""
            last_valid_candidate_for_env = ""

            while (not action_success) and inner_counter > 0:
                # Rebuild prompt each retry so the LLM can see rule feedback.
                action_prompt = _build_action_prompt()

                action_orig = llm(action_prompt, system=system_prompt, model=model).strip()
                action_orig = re.sub(r'^[^a-zA-Z]+', '', action_orig).strip()
                if action_orig.startswith('To solve'):
                    action_orig = 'think: ' + action_orig

                action_candidate, action_for_env_candidate = normalize_action_for_react(action_orig)

                # Think actions do not interact with the environment and do not need rule checks.
                if action_candidate.startswith('think:'):
                    action = action_candidate
                    action_for_env = ""
                    action_success = True
                    break

                # Parse latest textual observation into state JSON for rule checking.
                state_text = process_env_history(str(env_history))
                try:
                    state_json = state_info_transformation(state_text)
                except Exception:
                    state_json = {}
                if state_json is None:
                    state_json = {}

                # Convert action text into action JSON for rule checking.
                action_json = convert_action(action_candidate)
                if action_json is None:
                    imagination_attempt += 1
                    env_history.add("action", f"Action in Imagination (attempt {imagination_attempt}): {action_candidate}.")
                    env_history.add("observation", "Invalid action format. Please output a single valid action.")
                    inner_counter -= 1
                    continue
                last_valid_candidate = action_candidate
                last_valid_candidate_for_env = action_for_env_candidate

                # Rules-only feasibility check (no World Model).
                rule_check = buffer.worldcode_get_prediction(state_json, action_json, sg.graph)

                if rule_check.get('success'):
                    action = action_candidate
                    action_for_env = action_for_env_candidate
                    action_success = True
                    break

                imagination_attempt += 1
                env_history.add("action", f"Action in Imagination (attempt {imagination_attempt}): {action_candidate}.")
                observation = f"{rule_check.get('feedback', '')}. {rule_check.get('suggestion', '')}".strip()
                env_history.add("observation", observation)
                inner_counter -= 1

            if not action_success:
                # Give up after retries and execute the last sampled action anyway.
                action = last_valid_candidate or "look"
                action_for_env = last_valid_candidate_for_env

        if action.startswith('To solve'):
            action = 'think: ' + action

        env_history.add("action", action)
        
        # 处理 think 动作：不与环境交互
        if action.startswith('think:'):
            observation = 'OK.'
            done = False
            env_history.add("observation", observation)
        else:
            # 执行动作
            observation, reward, done, info = env.step([action_for_env or action])
            observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
            env_history.add("observation", observation)

            # 只有当动作是有效物理动作时，才更新轨迹和场景图
            for action_keyword in valid_actions:
                if action.startswith(action_keyword):
                    trajectory_records.append({"action": action, "observation": observation})
                    
                    # update scene graph
                    # ! [1. data collection] raw trajectory buffer path 
                    # ! [3. inference time] save sg continuously
                    sg_history.append(copy.deepcopy(sg.graph))
                    interaction_info = '> ' + action + '\n' + observation
                    sg.update_graph(interaction_info)
                    sg.save_to_json(sg_file_path)
                    break # 找到匹配的前缀后即可退出循环


        if to_print:
            print(f'> {action}\n{observation}')
            sys.stdout.flush()
        if done:
            return env_history, True, sg_history, progress_memory_log_events
        # if the action is the same as the previous action, terminate the interaction
        # elif env_history.check_is_exhausted():
        #     return env_history, False, sg_history
        cur_step += 1
    return env_history, False, sg_history, progress_memory_log_events

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
        progress_memory: bool = False,
        progress_memory_library_path: str = "",
        progress_memory_top_tasks: Optional[int] = None,
        use_local_fewshot: bool = False,
        feasibility_memory: bool = True,
        task_file: str = "",
        checkpoint_env_configs_path: str = "",
        checkpoint_every: int = 1,
        resume: bool = False,
    ) -> List[Dict[str, Any]]:
    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)

    # ! delete inferenceTime_SG at the beginning of each trial if it exists
    if trial_idx == 0 and not resume:
        inferenceTime_SG_dir = os.path.join(io_dir, 'traj_data', env_name, 'inferenceTime_SG')
        if os.path.exists(inferenceTime_SG_dir):
            for item in os.listdir(inferenceTime_SG_dir):
                item_path = os.path.join(inferenceTime_SG_dir, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except Exception as e:
                    print(f"Error deleting {item_path}: {e}")

    def _checkpoint_env_configs() -> None:
        if not checkpoint_env_configs_path:
            return
        try:
            tmp_path = checkpoint_env_configs_path + ".tmp"
            with open(tmp_path, "w") as wf:
                json.dump(env_configs, wf, indent=4)
            os.replace(tmp_path, checkpoint_env_configs_path)
        except Exception:
            # Best-effort only; never fail the run due to checkpoint I/O.
            return


    with open(_default_config_path()) as reader:
        config = yaml.safe_load(reader)
    split = "eval_out_of_distribution"

    if not task_file:
        raise ValueError(
            "An ordered ALFWorld task file is required for reproducible evaluation."
        )
    env = init_ordered_alfworld_env(
        config=config,
        environment_type=config["env"]["type"],
        split=split,
        task_file=task_file,
        expected_num_envs=len(env_configs),
    )

    progress_memory_planner = None
    if progress_memory:
        if not progress_memory_library_path:
            raise ValueError("--progress_memory requires --progress_memory_library produced by Stage 2.")
        if not os.path.isfile(progress_memory_library_path):
            raise FileNotFoundError(f"Progress memory library not found: {progress_memory_library_path}")
        progress_memory_model = get_api_model(
            "progress_memory_milestone_progress",
            get_api_model("progress_memory_planner", model),
        )
        kwargs: Dict[str, Any] = {
            "library_path": progress_memory_library_path,
            "prompt_dir": FOLDER,
            "model_name": progress_memory_model,
            "milestone_check_mode": "llm",
        }
        if progress_memory_top_tasks is not None:
            kwargs["top_tasks"] = int(progress_memory_top_tasks)
        progress_memory_planner = ProgressMemoryPlanner(**kwargs)
    if feasibility_memory and not buffer.functions_set:
        raise FileNotFoundError(
            "--feasibility_memory requires Stage 1 Feasibility Memory rules at "
            f"{buffer.rule_code_file}"
        )

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)
    
    # Statistics tracking per task type (6 categories)
    cnts = [0] * 6  # Count of attempts per category
    rs = [0] * 6    # Count of successes per category

    # 遍历task env
    for z, env_config in enumerate(env_configs):
        task_id = z
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        # 提取任务名
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])

        print(f"using {name}")

        if env_config.get("skip"):
            if env_config.get("is_success"):
                num_successes += 1

            for i, (k, v) in enumerate(PREFIXES.items()):
                if name.startswith(k):
                    cnts[i] += 1
                    if env_config.get("is_success"):
                        rs[i] += 1
                    break

            if checkpoint_env_configs_path and int(checkpoint_every) > 0 and ((z + 1) % int(checkpoint_every) == 0):
                _checkpoint_env_configs()
            continue

        if env_config["is_success"]:
            num_successes += 1
            
            # Track which category this success belongs to
            for i, (k, v) in enumerate(PREFIXES.items()):
                if name.startswith(k):
                    cnts[i] += 1
                    rs[i] += 1
                    break

            # log to world log
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            if checkpoint_env_configs_path and int(checkpoint_every) > 0 and ((z + 1) % int(checkpoint_every) == 0):
                _checkpoint_env_configs()
            continue
        

        for i, (k, v) in enumerate(PREFIXES.items()):
            if name.startswith(k):
                task_examples = d[f'react_{v}_1'] + d[f'react_{v}_0']
                # Few-shot examples go into the system prompt (align with WebShop s3_main).
                base_prompt = ''
                # 跑一个task env
                final_env_history, is_success, sg_history, progress_memory_log_events = alfworld_run(
                    env,
                    base_prompt,
                    taskID=task_id,
                    to_print=True,
                    ob=ob,
                    model=model,
                    progress_memory_planner=progress_memory_planner,
                    task_examples=task_examples,
                    trial_log_path=trial_log_path,
                    use_local_fewshot=use_local_fewshot,
                    feasibility_memory=feasibility_memory,
                )

                # update env config and statistics
                cnts[i] += 1
                if is_success:
                    status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
                    env_configs[z]['is_success'] = True
                    num_successes += 1
                    num_additional_successes += 1
                    rs[i] += 1
                else:
                    status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'

                # log to world log
                with open(world_log_path, 'a') as f:
                    f.write(status_str + '\n')

                # log env results to trial log
                with open(trial_log_path, 'a') as wf:
                    wf.write(f'\n#####\n\nEnvironment #{z}:\n')
                    wf.write(render_env_history_with_progress_memory_events(final_env_history, progress_memory_log_events))
                    wf.write(f'\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')
                
                print(f'look: i: {i}, k: {k}, v: {v}')
                
                # Log per-category statistics
                stats_info = f"{z+1} r {1 if is_success else 0} rs {rs} cnts {cnts} sum(rs)/sum(cnts) {sum(rs) / max(sum(cnts), 1)}"
                print(stats_info)
                with open(world_log_path, 'a') as f:
                    f.write(stats_info + '\n')
                with open(trial_log_path, 'a') as wf:
                    wf.write(stats_info + '\n')

                if checkpoint_env_configs_path and int(checkpoint_every) > 0 and ((z + 1) % int(checkpoint_every) == 0):
                    _checkpoint_env_configs()


                # ####################
                # # ! [1. data collection] raw trajectory buffer path 
                # file_path = os.path.join(io_dir, 'traj_data', env_name, 'buffer_traj', f'traj_{z}', f'transition_info_EnvironmentID{z}_Trial#{trial_idx}.json')

                # os.makedirs(os.path.dirname(file_path), exist_ok=True)
                # with open(file_path, 'w') as f:
                #     # json.dump(final_env_history, f, cls=NumpyEncoder, indent=4)
                #     f.write(str(final_env_history))
                # print(f'look: file_path: {file_path}')

                # sg_history_file = os.path.join(io_dir, 'traj_data', env_name, 'buffer_SG', f'traj_{z}', f'sg_transition_info_EnvironmentID{z}_Trial#{trial_idx}.json')
                # os.makedirs(os.path.dirname(sg_history_file), exist_ok=True)
                # # with open(sg_history_file, 'w') as f:
                # #     f.write(str(sg_history))
                # with open(sg_history_file, 'w') as f:
                #     json.dump(sg_history, f)
                # print(f'look: sg_history_file: {sg_history_file}')
                # ####################

        # ####################
        # # ! [1. data collection] raw trajectory --> json format
        # # TODO raw trajectory buffer path to json format
        # if (task_id + 1) % interval == 0 or (task_id + 1 == num_envs and (task_id + 1) % interval):
        #     if task_id + 1 == num_envs and (task_id + 1) % interval:
        #         current_interval = (task_id + 1) % interval
        #     else:
        #         current_interval = interval

        #     start_task_id = max(0, task_id + 1 - current_interval)

        #     ## Stage : transition buffering
        #     # ! [1. data collection] raw trajectory --> json format
        #     buffer.string_buffer_for_transitions_pure(current_interval, start_task_id)

        #     # TODO add updated rules mining component

        #     if task_id >= 100:
        #         ttd
        # ####################




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
