import os
import shutil
from typing import Any, Dict, List, Optional, Tuple
import json
import time
from api_config import apply_api_config, get_api_model
from buffer import Buffer, state_info_transformation
from env_history import EnvironmentHistory
from ruleminer import RuleMiner
from rulesverification import RuleVerifier
from trajectory_utils import render_env_history_for_buffer
from utils import chat_one_line
from webshop_env import WebShopEnv, build_state_from_env_session, parse_action
from task_file import validate_webshop_task
from webshop_sg import WebShopSceneGraph
from webshop_prompts import (
    build_react_action_system_prompt,
    build_react_action_user_prompt,
    postprocess_action_text,
)


S1_TRAJECTORY_MODEL = get_api_model("s1_trajectory", "gpt-4o-mini")


def _default_io_dir() -> str:
    env_dir = os.environ.get("DUALMEMORY_WEBSHOP_DIR")
    if env_dir:
        return env_dir
    return os.path.abspath(os.path.dirname(__file__))


def sample_action(prompt: str, system: str, model: str) -> str:
    action = chat_one_line(
        prompt=prompt,
        system=system,
        model=model,
        temperature=0.0,
        max_tokens=128,
        stop=["\n"],
    )
    return postprocess_action_text(action)


def webshop_run(
    env: WebShopEnv,
    session_id: str,
    base_prompt: str,
    model: str,
    max_steps: int = 15,
    feasibility_memory_during_collection: bool = False,
    rules_buffer: Optional[Buffer] = None,
    max_rule_retries: int = 5,
    show_progress: bool = True,
    expected_task_text: str = "",
) -> Tuple[EnvironmentHistory, bool, float]:
    reset_out = env.reset(session_id)
    ob = reset_out.observation
    if expected_task_text:
        validate_webshop_task(expected_task_text, ob)

    env_history = EnvironmentHistory(base_query="", start_info=ob, memory=[], history=[])
    env_history.reset()

    current_observation = ob
    reward = 0.0
    sg = WebShopSceneGraph()
    sg.update_from_env_session(env.sessions[session_id])
    sg_history: List[Dict[str, Any]] = []
    system_prompt = build_react_action_system_prompt()

    for step in range(max_steps):
        def _sample() -> str:
            user_prompt = build_react_action_user_prompt(
                init_prompt=base_prompt,
                first_observation=ob,
                steps=env_history.as_steps(),
            )
            return sample_action(user_prompt, system=system_prompt, model=model)
        # 得到action
        action = _sample()
        # 如果是online learning
        if feasibility_memory_during_collection and rules_buffer is not None and getattr(rules_buffer, "functions_set", []):
            inner = max(1, int(max_rule_retries))
            imagination_attempt = 0
            last_valid = action
            while inner > 0:
                act_json = parse_action(action)
                if act_json is None:
                    imagination_attempt += 1
                    env_history.add("action", f"Action in Imagination (attempt {imagination_attempt}): {action}.")
                    env_history.add("observation", "Invalid action format. Please output a single valid action.")
                    inner -= 1
                    action = _sample()
                    continue

                if act_json.get("name") == "think":
                    break

                state_json = build_state_from_env_session(env.sessions[session_id], current_observation)
                rule_check = rules_buffer.worldcode_get_prediction(
                    state_json,
                    act_json,
                    scene_graph=sg.snapshot(),
                )
                if rule_check.get("success"):
                    break

                imagination_attempt += 1
                env_history.add("action", f"Action in Imagination (attempt {imagination_attempt}): {action}.")
                feedback = str(rule_check.get("feedback", "")).strip()
                suggestion = str(rule_check.get("suggestion", "")).strip()
                env_history.add("observation", f"{feedback} {suggestion}".strip())
                inner -= 1
                last_valid = action
                action = _sample()

            if inner <= 0:
                action = last_valid or action

        if show_progress:
            page_type = env.sessions.get(session_id, {}).get("page_type", "unknown")
            print(f"[S1] {session_id} step {step + 1}/{max_steps} page={page_type} action={action}", flush=True)

        env_history.add("action", action)
        # 将action 结构化
        act_json = parse_action(action)
        if act_json is not None and act_json.get("name") != "think":
            sg_history.append(sg.snapshot())
        # 执行action
        step_out = env.step(session_id, action)
        env_history.add("observation", step_out.observation)
        # 如果动作成功
        if step_out.action_success:
            current_observation = step_out.observation
            # 告诉场景图刚刚的动作成功了
            sg.record_action_outcome(action_text=action, success=True)
            # 解析新的Session状态
            sg.update_from_env_session(env.sessions[session_id])
        else:
            # 告诉场景图刚刚的动作失败了
            sg.record_action_outcome(action_text=action, success=False)
        reward = float(step_out.reward)

        # 如果任务结束
        if step_out.done:
            is_success = reward >= 1.0 - 1e-6
            # attach sg_history to env_history object for caller to save (avoid API change)
            setattr(env_history, "_sg_history", sg_history)
            if show_progress:
                print(
                    f"[S1] {session_id} DONE success={is_success} reward={reward:.2f} steps={len(env_history.as_steps())}",
                    flush=True,
                )
            return env_history, is_success, reward

    setattr(env_history, "_sg_history", sg_history)
    if show_progress:
        print(f"[S1] {session_id} STOP max_steps={max_steps} reward={reward:.2f}", flush=True)
    return env_history, False, reward


def run_trial(
    trial_log_path: str,
    world_log_path: str,
    trial_idx: int,
    env_configs: List[Dict[str, Any]],
    model: str = S1_TRAJECTORY_MODEL,
    defer_rules: bool = False,
    online_rules: bool = False,
    interval: int = 10,
    max_steps: int = 15,
    base_prompt: str = "",
    task_texts: Optional[Dict[int, str]] = None,
    session_ids: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    apply_api_config()

    io_dir = _default_io_dir()
    env_name = "webshop"

    env = WebShopEnv()
    buffer = Buffer(io_dir=io_dir, env_name=env_name)
    miner = RuleMiner(io_dir=io_dir, env_name=env_name)
    verifier = RuleVerifier(env_name=env_name, io_dir=io_dir)

    if online_rules:
        buffer.functions_set = []
        print("[OnlineRules] Enabled: starting from empty rules.")

    # clear buffer_traj
    # 清除旧数据
    buffer_traj_dir = os.path.join(io_dir, "traj_data", env_name, "buffer_traj")
    if os.path.exists(buffer_traj_dir):
        for item in os.listdir(buffer_traj_dir):
            path = os.path.join(buffer_traj_dir, item)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except OSError:
                    pass

    num_envs = len(env_configs)
    num_successes = 0
    num_additional_successes = 0

    # 循环task
    for z, env_config in enumerate(env_configs):
        start_t = time.time()
        task_id = int(env_config["task_id"])
        session_id = (session_ids or {}).get(task_id, f"fixed_{task_id}")

        # 如果task之前已经成功过，跳过
        if env_config.get("is_success"):
            num_successes += 1
            print(f"[S1] Trial {trial_idx} env {z + 1}/{num_envs} task={task_id} SKIP (already success)", flush=True)
            with open(world_log_path, "a", encoding="utf-8") as wf:
                wf.write(f"Task #{task_id} Trial #{trial_idx}: SUCCESS (skipped)\n")
            continue
        
        # 跑一个task
        print(f"[S1] Trial {trial_idx} env {z + 1}/{num_envs} task={task_id} START", flush=True)
        final_env_history, is_success, reward = webshop_run(
            env=env,
            session_id=session_id,
            base_prompt=base_prompt,
            model=model,
            max_steps=max_steps,
            feasibility_memory_during_collection=online_rules,
            rules_buffer=buffer,
            show_progress=True,
            expected_task_text=(task_texts or {}).get(task_id, ""),
        )

        if is_success:
            status_str = f"Task #{task_id} Trial #{trial_idx}: SUCCESS reward={reward:.2f}"
            env_config["is_success"] = True
            num_successes += 1
            num_additional_successes += 1
        else:
            status_str = f"Task #{task_id} Trial #{trial_idx}: FAIL reward={reward:.2f}"

        elapsed_s = time.time() - start_t
        print(
            f"[S1] Trial {trial_idx} env {z + 1}/{num_envs} task={task_id} "
            f"{'SUCCESS' if is_success else 'FAIL'} reward={reward:.2f} time={elapsed_s:.1f}s",
            flush=True,
        )

        with open(world_log_path, "a", encoding="utf-8") as wf:
            wf.write(status_str + "\n")

        with open(trial_log_path, "a", encoding="utf-8") as wf:
            wf.write(
                f"\n#####\n\nEnvironment #{task_id}:\n{str(final_env_history)}\n\nSTATUS: {'OK' if is_success else 'FAIL'}\n\n#####\n"
            )

        file_path = os.path.join(
            io_dir,
            "traj_data",
            env_name,
            "buffer_traj",
            f"traj_{task_id}",
            f"transition_info_TaskID{task_id}_Trial#{trial_idx}.json",
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(render_env_history_for_buffer(final_env_history))

        sg_history = getattr(final_env_history, "_sg_history", []) or []
        sg_history_file = os.path.join(
            io_dir,
            "traj_data",
            env_name,
            "buffer_SG",
            f"traj_{task_id}",
            "sg_" + os.path.basename(file_path),
        )
        os.makedirs(os.path.dirname(sg_history_file), exist_ok=True)
        with open(sg_history_file, "w", encoding="utf-8") as f:
            json.dump(sg_history, f, indent=2)

        # rule mining/verification every interval tasks
        # 规则部分
        if not defer_rules:
            cur_idx = z + 1
            # 每interval个task触发一次
            if cur_idx % interval == 0 or (cur_idx == num_envs and (cur_idx % interval)):
                current_interval = interval if (cur_idx % interval == 0) else (cur_idx % interval)
                start_id = task_id + 1 - current_interval
                print(
                    f"[S1] Rule mining/verifying interval={current_interval} tasks={start_id}..{task_id}",
                    flush=True,
                )
                # 将轨迹转化为action observation pair，并分为集合
                buffer.string_buffer_for_transitions_pure(current_interval, start_id, cleanup=False)
                # 规则挖掘
                miner.get_rules_all()
                # 生成规则代码
                verifier.rules_code_all()
                # 验证规则
                verifier.functions_verification()
                # 选择规则
                verifier.select_rules()

                if online_rules:
                    buffer.functions_set = []
                    if os.path.exists(buffer.rule_code_file):
                        buffer.load_functions_from_file(buffer.rule_code_file)
                    print(f"[OnlineRules] Reloaded rules: {len(buffer.functions_set)}")

    log_str = (
        f"\n-----\nSUCCESS: {num_successes}\nADDITIONAL SUCCESS: {num_additional_successes}\n"
        f"FAIL: {num_envs - num_successes}\nTOTAL: {num_envs}\n"
        f"ACCURACY: {round(num_successes / max(1, num_envs), 2)}\n-----\n"
    )
    with open(trial_log_path, "a", encoding="utf-8") as wf:
        wf.write(log_str)
    with open(world_log_path, "a", encoding="utf-8") as wf:
        wf.write(log_str + "\n")

    return env_configs
