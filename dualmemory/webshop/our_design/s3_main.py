import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from api_config import apply_api_config, get_api_model
from buffer import Buffer
from env_history import EnvironmentHistory
from progress_memory import ProgressMemoryPlanner, build_action_prompt, extract_instruction
from llm_metrics import get_metrics, reset_metrics, write_metrics
from utils import chat_one_line
from webshop_env import WebShopEnv, build_state_from_env_session, parse_action
from task_file import load_webshop_tasks, validate_webshop_task
from webshop_sg import WebShopSceneGraph
from webshop_prompts import (
    build_react_action_system_prompt,
    build_react_action_user_prompt,
    postprocess_action_text,
)


ACTION_MODEL = get_api_model("s3_action", get_api_model("s1_trajectory", "gpt-4o-mini"))
PLANNER_MODEL = get_api_model("progress_memory_planner", "gpt-4o-mini")


def _default_io_dir() -> str:
    env_dir = os.environ.get("DUALMEMORY_WEBSHOP_DIR")
    if env_dir:
        return env_dir
    return os.path.abspath(os.path.dirname(__file__))


def summarize_state(state: Dict[str, Any], observation: str, max_obs_chars: int = 800) -> str:
    lines = [f"page_type: {state.get('page_type', 'unknown')}"]
    if state.get("query_string"):
        lines.append(f"query: {state.get('query_string')} (page {state.get('page_num', 1)})")
    if state.get("asin"):
        lines.append(f"asin: {state.get('asin')}")
    if state.get("options"):
        lines.append(f"selected_options: {state.get('options')}")
    obs = (observation or "").strip()
    if obs:
        snippet = obs[:max_obs_chars]
        lines.append("observation_snippet:")
        lines.append(snippet)
    return "\n".join(lines)


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

def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_episode(
    *,
    env: WebShopEnv,
    session_id: str,
    task_id: int,
    action_model: str,
    planner: ProgressMemoryPlanner,
    buffer: Buffer,
    progress_memory: bool,
    feasibility_memory: bool,
    max_rule_retries: int,
    max_steps: int,
    show_progress: bool,
    prompt_dir: str,
    base_prompt: str,
    retrieval_log_path: str | None,
    expected_task_text: str = "",
) -> Tuple[EnvironmentHistory, bool, float, List[str]]:
    reset_out = env.reset(session_id)
    ob = reset_out.observation

    task_text = extract_instruction(ob)
    if expected_task_text:
        validate_webshop_task(expected_task_text, ob)
    # 构建当前task的milestone
    milestone_guide: List[str] = (
        planner.build_milestone_guide(task_text) if progress_memory and planner.has_library() else []
    )
    current_milestone_idx = 0

    env_history = EnvironmentHistory(base_query="", start_info=ob, memory=[], history=[])
    env_history.reset()
    if progress_memory and milestone_guide:
        guide_examples = getattr(planner, "last_guide_examples", []) or []
        if guide_examples:
            lines = ["[PROGRESS_MEMORY] milestone_guide: similar tasks used as examples:"]
            for ex in guide_examples:
                tid = ex.get("task_id")
                sim = ex.get("similarity")
                task = str(ex.get("task", "")).strip().replace("\n", " ")
                sim_str = f"{float(sim):.4f}" if isinstance(sim, (int, float)) else "N/A"
                lines.append(f"[PROGRESS_MEMORY] - task_id={tid} similarity={sim_str} task={task}")
            env_history.add("human_edit", "\n".join(lines))
            if retrieval_log_path:
                _append_jsonl(
                    retrieval_log_path,
                    {
                        "event": "milestone_guide_examples",
                        "session_id": session_id,
                        "task_id": task_id,
                        "examples": guide_examples,
                    },
                )

    current_observation = ob
    reward = 0.0
    sg = WebShopSceneGraph()
    sg.update_from_env_session(env.sessions[session_id])
    if progress_memory:
        init_prompt = (base_prompt or "").strip()
        system_prompt = build_react_action_system_prompt()
        if init_prompt:
            system_prompt = f"{system_prompt}\n{init_prompt}"
    else:
        system_prompt = build_react_action_system_prompt()

    # 最多15步
    if show_progress:
        mode = "PROGRESS_MEMORY" if progress_memory else "ReAct"
        rules_active = bool(feasibility_memory)
        print(
            f"[S3] {session_id} task={task_id} mode={mode} "
            f"rules={'on' if rules_active else 'off'}",
            flush=True,
        )

    last_logged_milestone: str | None = None
    for step in range(max_steps):
        state = build_state_from_env_session(env.sessions[session_id], current_observation)
        state["raw_observation"] = current_observation

        if progress_memory:
            # 目前的轨迹，过滤了Action in Imagination
            recent_traj = [
                s for s in env_history.as_steps() if not str(s.get("action", "")).startswith("Action in Imagination")
            ]
            # 确定当前milestone
            if milestone_guide:
                current_milestone_idx = planner.determine_current_milestone_idx(
                    task_text=task_text,
                    milestone_guide=milestone_guide,
                    current_milestone_idx=current_milestone_idx,
                    recent_trajectory=recent_traj,
                )
                current_milestone = milestone_guide[current_milestone_idx]
            else:
                current_milestone = ""

            if current_milestone:
                milestone_demos, milestone_demos_meta = planner.retrieve_milestone_demos_with_meta(
                    current_milestone,
                    exclude_task_ids=None,
                    mask=True,
                )
            else:
                milestone_demos, milestone_demos_meta = "- None", []
        else:
            current_milestone = ""
            milestone_demos, milestone_demos_meta = "- None", []

        if (
            progress_memory
            and milestone_guide
            and current_milestone
            and current_milestone != last_logged_milestone
        ):
            ms = f"{current_milestone_idx + 1}/{len(milestone_guide)}"
            lines = [f"[PROGRESS_MEMORY] retrieved milestone few-shot for milestone {ms}:"]
            lines.append(f"[PROGRESS_MEMORY] query_milestone: {current_milestone}")
            if milestone_demos_meta:
                lines.append("[PROGRESS_MEMORY] retrieved_from_library:")
                for m in milestone_demos_meta:
                    sim = m.get("similarity")
                    sim_str = f"{float(sim):.4f}" if isinstance(sim, (int, float)) else "N/A"
                    lines.append(
                        "[PROGRESS_MEMORY] - rank={rank} task_id={task_id} milestone_id={milestone_id} "
                        "similarity={sim} milestone={milestone}".format(
                            rank=m.get("rank"),
                            task_id=m.get("task_id"),
                            milestone_id=m.get("milestone_id"),
                            sim=sim_str,
                            milestone=str(m.get("milestone", "")).strip().replace("\n", " "),
                        )
                    )
            else:
                lines.append("[PROGRESS_MEMORY] retrieved_from_library: - None")
            lines.append("[PROGRESS_MEMORY] milestone_demos_text (exactly as inserted into prompt):")
            lines.append("[PROGRESS_MEMORY] ---BEGIN_DEMOS---")
            lines.append(str(milestone_demos or "- None"))
            lines.append("[PROGRESS_MEMORY] ---END_DEMOS---")
            env_history.add("human_edit", "\n".join(lines))
            last_logged_milestone = current_milestone

            if retrieval_log_path:
                _append_jsonl(
                    retrieval_log_path,
                    {
                        "event": "milestone_fewshot_retrieval",
                        "session_id": session_id,
                        "task_id": task_id,
                        "step": step + 1,
                        "milestone_idx": current_milestone_idx,
                        "milestone": current_milestone,
                        "progress_check": getattr(planner, "last_progress_check", {}) or {},
                        "retrieved": milestone_demos_meta,
                        "demos_text": milestone_demos,
                    },
                )

        # 构建prompt（PROGRESS_MEMORY 或 ReAct-style history only）
        def _make_prompt() -> str:
            if not progress_memory:
                return build_react_action_user_prompt(
                    init_prompt=base_prompt,
                    first_observation=ob,
                    steps=env_history.as_steps(),
                )
            return build_action_prompt(
                prompt_dir=prompt_dir,
                milestone_guide=milestone_guide,
                current_milestone=current_milestone,
                milestone_demos=milestone_demos,
                history_text=env_history.to_prompt_text(include_human_edit=False),
            )
        # 得到action
        action = sample_action(_make_prompt(), system=system_prompt, model=action_model)

        if show_progress:
            page_type = state.get("page_type", "unknown")
            if progress_memory and milestone_guide:
                ms = f"{current_milestone_idx + 1}/{len(milestone_guide)}"
                print(
                    f"[S3] {session_id} step {step + 1}/{max_steps} page={page_type} ms={ms} action={action}",
                    flush=True,
                )
            else:
                print(
                    f"[S3] {session_id} step {step + 1}/{max_steps} page={page_type} action={action}",
                    flush=True,
                )

        if feasibility_memory and buffer.functions_set:
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
                    action = sample_action(_make_prompt(), system=system_prompt, model=action_model)
                    continue

                if act_json.get("name") == "think":
                    break
                
                # 过一遍规则
                rule_check = buffer.worldcode_get_prediction(state, act_json, scene_graph=sg.snapshot())
                if rule_check.get("success"):
                    break

                imagination_attempt += 1
                env_history.add("action", f"Action in Imagination (attempt {imagination_attempt}): {action}.")
                feedback = str(rule_check.get("feedback", "")).strip()
                suggestion = str(rule_check.get("suggestion", "")).strip()
                env_history.add("observation", f"{feedback} {suggestion}".strip())
                inner -= 1
                last_valid = action
                action = sample_action(_make_prompt(), system=system_prompt, model=action_model)

            if inner <= 0:
                action = last_valid or action

        if progress_memory and milestone_guide and current_milestone:
            env_history.add(
                "human_edit",
                f"[PROGRESS_MEMORY] milestone {current_milestone_idx + 1}/{len(milestone_guide)}: {current_milestone}",
            )

        env_history.add("action", action)
        act_json = parse_action(action)
        if act_json is not None and act_json.get("name") != "think":
            # Provide sg snapshot BEFORE executing the action (same convention as ALFWorld buffering).
            sg_snapshot = sg.snapshot()
        else:
            sg_snapshot = None
        # 实施动作
        step_out = env.step(session_id, action)
        env_history.add("observation", step_out.observation)
        if step_out.action_success:
            current_observation = step_out.observation
            if sg_snapshot is not None:
                sg.record_action_outcome(action_text=action, success=True)
                sg.update_from_env_session(env.sessions[session_id])
        reward = float(step_out.reward)

        if step_out.done:
            return env_history, (reward >= 1.0 - 1e-6), reward, milestone_guide

    return env_history, False, reward, milestone_guide


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S3: WebShop inference with progress memory + feasibility memory.")
    parser.add_argument("--run_name", type=str, default="s3_webshop_run")
    parser.add_argument("--task_file", type=str, required=True, help="Published ordered WebShop task JSON.")
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--model", type=str, default=ACTION_MODEL)
    parser.add_argument("--planner_model", type=str, default=PLANNER_MODEL)
    parser.add_argument("--progress_memory_library", type=str, default="")
    parser.add_argument("--embedding_model", type=str, default=get_api_model("embedding", "all-mpnet-base-v2"))
    parser.add_argument("--top_tasks", type=int, default=3)
    parser.add_argument("--top_milestones", type=int, default=3)
    parser.add_argument(
        "--progress_memory",
        action="store_true",
        help="Enable PROGRESS_MEMORY (milestones + retrieved demos) for action generation. "
        "If not set, generate actions from history only (ReAct-style).",
    )
    parser.add_argument("--feasibility_memory", action="store_true")
    parser.add_argument("--max_rule_retries", type=int, default=5)
    parser.add_argument("--no_progress", action="store_true", help="Disable per-step progress printing.")
    parser.add_argument(
        "--base_prompt_file",
        type=str,
        default="",
        help="Optional few-shot prompt file (same style as S1). "
        "Default: prompts/s1_react_prompt1.txt when not using PROGRESS_MEMORY; "
        "prompts/s1_default_fewshot.txt when using PROGRESS_MEMORY.",
    )
    return parser.parse_args()


def _resolve_run_dir(run_name: str) -> str:
    if not run_name:
        raise ValueError("run_name must be non-empty")
    if os.path.isabs(run_name):
        return run_name
    base_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_dir, run_name)


def main() -> None:
    args = get_args()
    reset_metrics()
    apply_api_config()

    tasks = load_webshop_tasks(args.task_file)

    io_dir = _default_io_dir()
    prompt_dir = os.path.join(io_dir, "prompts")
    env_name = "webshop"
    base_prompt = ""
    if args.base_prompt_file:
        base_prompt_path = args.base_prompt_file
    elif args.progress_memory:
        base_prompt_path = os.path.join(prompt_dir, "s1_react_prompt1.txt")
    else:
        base_prompt_path = os.path.join(prompt_dir, "s1_react_prompt1.txt")
    if base_prompt_path and os.path.exists(base_prompt_path):
        with open(base_prompt_path, "r", encoding="utf-8") as f:
            base_prompt = f.read().rstrip() + "\n"

    if args.progress_memory and not args.progress_memory_library:
        raise ValueError("--progress_memory requires --progress_memory_library produced by Stage 2.")
    if args.progress_memory and not os.path.isfile(args.progress_memory_library):
        raise FileNotFoundError(f"Progress memory library not found: {args.progress_memory_library}")

    env = WebShopEnv()
    buffer = Buffer(io_dir=io_dir, env_name=env_name, model_name=args.model)
    if args.feasibility_memory and not buffer.functions_set:
        raise FileNotFoundError(
            "--feasibility_memory requires Stage 1 rules at "
            f"{buffer.rule_code_file}"
        )
    planner = ProgressMemoryPlanner(
        library_path=args.progress_memory_library,
        prompt_dir=prompt_dir,
        model_name=args.planner_model,
        embedding_model=args.embedding_model,
        top_tasks=args.top_tasks,
        top_milestones=args.top_milestones,
    )

    logging_dir = _resolve_run_dir(args.run_name)
    os.makedirs(logging_dir, exist_ok=True)
    world_log_path = os.path.join(logging_dir, "world.log")
    trial_log_path = os.path.join(logging_dir, "trial_0.log")
    retrieval_log_path = os.path.join(logging_dir, "progress_memory_retrievals_trial_0.jsonl")
    if os.path.exists(trial_log_path):
        open(trial_log_path, "w").close()
    if os.path.exists(retrieval_log_path):
        open(retrieval_log_path, "w").close()

    with open(trial_log_path, "a", encoding="utf-8") as wf:
        wf.write("[S3] WebShop run header\n")
        wf.write(f"[S3] started_at: {datetime.now().isoformat(timespec='seconds')}\n")
        wf.write("[S3] logs:\n")
        wf.write(f"- trial_log: {trial_log_path}\n")
        wf.write(f"- world_log: {world_log_path}\n")
        wf.write(f"- progress_memory_retrievals_jsonl: {retrieval_log_path}\n")
        wf.write("[S3] config:\n")
        wf.write(
            json.dumps(
                {
                    "run_name": args.run_name,
                    "task_file": os.path.abspath(args.task_file),
                    "task_count": len(tasks),
                    "task_ids": [task["task_id"] for task in tasks],
                    "max_steps": args.max_steps,
                    "model": args.model,
                    "planner_model": args.planner_model,
                    "progress_memory": bool(args.progress_memory),
                    "feasibility_memory": bool(args.feasibility_memory),
                    "effective_rules": bool(args.feasibility_memory),
                    "max_rule_retries": args.max_rule_retries,
                    "embedding_model": args.embedding_model,
                    "top_tasks": args.top_tasks,
                    "top_milestones": args.top_milestones,
                    "progress_memory_library": args.progress_memory_library,
                    "base_prompt_file": base_prompt_path,
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )

    env_configs: List[Dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        env_configs.append({"task_id": task["task_id"], "is_success": False})

    successes = 0
    rewards: List[float] = []
    for task_index, task in enumerate(tasks):
        task_id = task["task_id"]
        session_id = task["session_idx"]
        env_history, ok, reward, guide = run_episode(
            env=env,
            session_id=session_id,
            task_id=task_id,
            action_model=args.model,
            planner=planner,
            buffer=buffer,
            progress_memory=args.progress_memory,
            feasibility_memory=args.feasibility_memory,
            max_rule_retries=args.max_rule_retries,
            max_steps=args.max_steps,
            show_progress=not bool(args.no_progress),
            prompt_dir=prompt_dir,
            base_prompt=base_prompt,
            retrieval_log_path=retrieval_log_path if args.progress_memory else None,
            expected_task_text=task["task_text"],
        )
        rewards.append(reward)
        successes += int(ok)
        env_configs[task_index]["is_success"] = bool(ok)

        with open(world_log_path, "a", encoding="utf-8") as wf:
            wf.write(f"Task #{task_id}: {'SUCCESS' if ok else 'FAIL'} reward={reward:.2f}\n")
        with open(trial_log_path, "a", encoding="utf-8") as wf:
            wf.write(
                f"\n#####\n\nEnvironment #{task_id}:\n{str(env_history)}\n\nSTATUS: {'OK' if ok else 'FAIL'}\n\n#####\n"
            )
            if guide:
                wf.write("\n[PROGRESS_MEMORY] milestone_guide:\n" + "\n".join(f"- {m}" for m in guide) + "\n")

    summary = {
        "task_file": os.path.abspath(args.task_file),
        "task_count": len(tasks),
        "task_ids": [task["task_id"] for task in tasks],
        "success_rate": successes / max(1, len(tasks)),
        "avg_reward": sum(rewards) / max(1, len(rewards)),
    }
    metrics = get_metrics()
    summary["llm_metrics"] = {
        k: metrics[k]
        for k in (
            "llm_call_count",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "latency_seconds",
            "avg_latency_seconds",
        )
    }
    with open(os.path.join(logging_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(logging_dir, "env_results_trial_0.json"), "w", encoding="utf-8") as f:
        json.dump(env_configs, f, indent=2)
    metrics_path = os.path.join(logging_dir, "llm_metrics.json")
    write_metrics(
        metrics_path,
        extra={
            "run_name": args.run_name,
            "task_file": os.path.abspath(args.task_file),
            "task_count": len(tasks),
            "model": args.model,
        },
    )
    with open(world_log_path, "a", encoding="utf-8") as wf:
        wf.write(
            "LLM_METRICS: "
            + json.dumps(summary["llm_metrics"], ensure_ascii=False)
            + f"\nLLM_METRICS_PATH: {metrics_path}\n"
        )


if __name__ == "__main__":
    main()
