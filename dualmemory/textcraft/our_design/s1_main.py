from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from api_config import apply_api_config, get_api_model
from buffer import Buffer
from llm_client import LLMConfig, make_chat_llm, make_openai_client
from parsing import (
    infer_action_success,
    is_think,
    parse_action,
    parse_goal,
    parse_inventory_observation,
    parse_recipes,
    to_action_json,
)
from prompts import REACT_SYSTEM_PROMPT, append_react_history, build_react_user_prompt
from rule_miner import RuleMiner
from rule_verifier import RuleVerifier
from rules_engine import RulesEngine
from textcraft_env import get_textcraft_env
from task_file import load_textcraft_tasks, validate_textcraft_task
from utils import RetryConfig, ensure_dir, write_json, write_text


def _default_io_dir() -> str:
    # Default to this directory (textcraft/our_design).
    return str(Path(__file__).resolve().parent)


def _console(msg: str = "") -> None:
    print(msg, flush=True)


def run_episode(
    env,
    *,
    seed: int,
    llm,
    max_steps: int,
    verbose: bool,
    rules: RulesEngine | None,
    max_rule_retries: int,
    save_dir: Path,
    expected_task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    obs, _info = env.reset(seed=seed)
    problem = str(obs).strip()
    if expected_task:
        validate_textcraft_task(expected_task, problem)

    _console(f"\n{'=' * 40}")
    _console(f"Episode {seed}")
    _console(f"Task: {problem}")
    _console("Action: reset")
    _console(f"Observation: {problem}\n")

    prompt_history = ""
    log_trajectory = f"Action: reset\nObservation: {problem}\n\n"

    steps: list[dict[str, Any]] = []

    # 只有当开启了--online_rules 并且 rules 不为空时，才使用 rule gating
    use_rule_gating = rules is not None and bool(rules.functions)
    # 当前这一局的最终目标，例如{'item': 'dark oak sign', 'count': 1}
    goal = None
    # 当前这一局用的合成配方，[{'output': {'item': 'planks', 'count': 4}, 'inputs': [...]}, ...]
    recipes = None
    # 所有可以通过合成得到的物品列表，['planks', 'stick', 'sign', ...]
    craftable_items: list[str] | None = None
    # 当前这一局的物品栏，{'planks': 4, 'stick': 2, 'sign': 1}
    inventory: dict[str, int] = {}
    # 我们是否已经确认过背包的内容
    inventory_known = True  # TextCraft 开局背包就是空的，直接从第 0 步开始追踪
    if use_rule_gating:
        goal = parse_goal(problem) or {"item": "", "count": 1}
        recipes = parse_recipes(problem)
        craftable_items = sorted(
            {
                r.get("output", {}).get("item")
                for r in (recipes or [])
                if isinstance(r, dict) and isinstance(r.get("output", {}).get("item"), str)
            }
        )

    # 维护agent内部状态，主要是inventory
    def _append_step(action: str, observation: str, reward: float, done: bool) -> None:
        nonlocal prompt_history, log_trajectory, inventory, inventory_known
        observation = str(observation).strip()
        action = str(action).strip()
        action_success = infer_action_success(action, observation)
        steps.append(
            {
                "action": action,
                "observation": observation,
                "reward": float(reward or 0.0),
                "done": bool(done),
                "action_success": bool(action_success),
            }
        )
        prompt_history = append_react_history(prompt_history, action, observation)
        log_trajectory += f"Action: {action}\nObservation: {observation}\n\n"

        if use_rule_gating:
            action_json = to_action_json(action) or {}
            name = action_json.get("name")
            # 如果动作时inventory，直接用obs替换维护的inventory
            if name == "inventory":
                inv = parse_inventory_observation(observation)
                if inv is not None:
                    inventory = dict(inv)
                    inventory_known = True
            # 如果action是get且成功，维护的inventory增加
            elif bool(action_success) and name == "get":
                args = action_json.get("args") or {}
                item = args.get("item")
                count = args.get("count")
                if isinstance(item, str) and isinstance(count, int):
                    inventory[item] = int(inventory.get(item, 0)) + int(count)
            # 如果action是craft且成功，维护的inventory增加产物，减少材料
            elif bool(action_success) and name == "craft":
                args = action_json.get("args") or {}
                out_item = args.get("item")
                out_count = args.get("count")
                inputs = args.get("inputs") or []

                if isinstance(inputs, list):
                    for ing in inputs:
                        if not isinstance(ing, dict):
                            continue
                        ing_item = ing.get("item")
                        ing_count = ing.get("count")
                        if isinstance(ing_item, str) and isinstance(ing_count, int):
                            inventory[ing_item] = int(inventory.get(ing_item, 0)) - int(ing_count)
                            if inventory[ing_item] <= 0:
                                inventory.pop(ing_item, None)

                if isinstance(out_item, str) and isinstance(out_count, int):
                    inventory[out_item] = int(inventory.get(out_item, 0)) + int(out_count)

        return None

    reward = 0.0
    done = False
    llm_calls = 0
    candidate_raw = ""
    max_patience = 8
    patience_ctr = 0

    terminate_reason = "max_steps"

    for _step_idx in range(1, max_steps + 1):
        if done:
            break

        # Sample an action (optionally with rule-gated resampling).
        candidate_action = ""

        def _sample() -> str:
            nonlocal candidate_raw, llm_calls
            candidate_raw = llm(build_react_user_prompt(problem, prompt_history), stop=["\n"])
            llm_calls += 1
            return parse_action(candidate_raw) or "inventory"

        # 采样一个动作
        candidate_action = _sample()

        # 如果是online learning
        if use_rule_gating and rules is not None and rules.functions:
            retries_left = max(0, int(max_rule_retries))
            imagination_attempt = 0
            last_action = candidate_action

            while retries_left > 0:
                if is_think(candidate_action) or candidate_action == "inventory":
                    break
                action_json = to_action_json(candidate_action)
                # 如果不属于get craft think inventory，则重新采样
                if not action_json or action_json.get("name") not in {"get", "craft"}:
                    imagination_attempt += 1
                    _console(
                        f"[RuleGating][seed={seed} step={_step_idx} attempt={imagination_attempt}] "
                        f"invalid action format: {candidate_action}"
                    )
                    prompt_history = append_react_history(
                        prompt_history,
                        f"Action in Imagination (attempt {imagination_attempt}): {candidate_action}",
                        "Invalid action format. Output one valid action.",
                    )
                    retries_left -= 1
                    candidate_action = _sample()
                    continue

                # 构建当前state
                state = {
                    "goal": goal or {"item": "", "count": 1},
                    "recipes": recipes or [],
                    "craftable_items": list(craftable_items or []),
                    "inventory": dict(inventory),
                }
                res = rules.check(state=state, action=action_json)
                if res.success:
                    break

                imagination_attempt += 1
                last_action = candidate_action
                _console(
                    f"[RuleGating][seed={seed} step={_step_idx} attempt={imagination_attempt}] "
                    f"blocked action: {candidate_action}"
                )
                if res.rule_id:
                    _console(f"[RuleGating] rule_id={res.rule_id}")
                if res.feedback:
                    _console(f"[RuleGating] feedback: {res.feedback}")
                if res.suggestion:
                    _console(f"[RuleGating] suggestion: {res.suggestion}")
                prompt_history = append_react_history(
                    prompt_history,
                    f"Action in Imagination (attempt {imagination_attempt}): {candidate_action}",
                    (res.feedback + " " + res.suggestion).strip(),
                )
                retries_left -= 1
                candidate_action = _sample()

            if retries_left <= 0:
                candidate_action = last_action

        action = candidate_action

        if "task completed" in action.lower():
            reward = 1.0
            done = True
            terminate_reason = "task_completed_thought"
            _console(f"Action: {action}\n")
            break
        if "task failed" in action.lower():
            terminate_reason = "task_failed_thought"
            _console(f"Action: {action}\n")
            break

        # Execute. 执行动作
        if is_think(action):
            observation = "OK."
            step_reward = 0.0
            step_done = False
        else:
            try:
                observation, step_reward, step_done, _truncated, _step_info = env.step(action)
            except Exception as e:
                observation = f"Invalid action: {e}"
                step_reward = 0.0
                step_done = False

        observation = str(observation).strip()
        # 记录本步，更新维护的inventory
        _append_step(action, observation, float(step_reward or 0.0), bool(step_done))

        reward = float(step_reward or 0.0)
        done = bool(step_done)

        _console(f"Action: {action}")
        _console(f"Observation: {observation}\n")

        if not is_think(action):
            if observation.startswith("Could not") or observation == "OK.":
                patience_ctr += 1
                if patience_ctr >= max_patience:
                    terminate_reason = "max_patience"
                    _console(f"Max patience ({max_patience}) reached at step {_step_idx}, terminating.")
                    break
            else:
                patience_ctr = 0

        if reward > 0 or done:
            terminate_reason = "done"
            break

    # Save
    ensure_dir(save_dir)
    log_path = save_dir / "episode.log"
    json_path = save_dir / "episode.json"
    write_text(log_path, log_trajectory.rstrip() + "\n")
    write_json(
        json_path,
        {
            "seed": seed,
            "problem": problem,
            "success": bool(reward > 0),
            "reward": reward,
            "done": done,
            "steps": steps,
            "llm_calls": llm_calls,
            "terminate_reason": terminate_reason,
            "raw_llm_last": candidate_raw,
        },
        indent=2,
    )

    _console(
        f"[EpisodeSummary] seed={seed} success={bool(reward > 0)} reward={reward:.1f} "
        f"steps={len(steps)} llm_calls={llm_calls} terminate_reason={terminate_reason}"
    )
    _console(f"[EpisodeSummary] saved to {save_dir}")

    return {
        "seed": seed,
        "success": bool(reward > 0),
        "reward": reward,
        "done": done,
        "steps": len(steps),
        "llm_calls": llm_calls,
        "terminate_reason": terminate_reason,
        "problem": problem,
        "trajectory": log_trajectory.rstrip() + "\n",
        "episode_dir": str(save_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: collect TextCraft trajectories (ReAct-style).")
    parser.add_argument("--task_file", type=str, required=True, help="Published ordered TextCraft task JSON.")
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--io_dir", type=str, default="")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--traj_dir", type=str, default="")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--rules_code_path", type=str, default="", help="Path to pruned_rules_code.json (optional).")
    parser.add_argument("--max_rule_retries", type=int, default=5, help="How many resamples allowed when a rule rejects the action.")
    parser.add_argument(
        "--online_rules",
        action="store_true",
        help="Use current mined rules during collection (starts empty; reloads every interval).",
    )
    parser.add_argument(
        "--rules_interval",
        "--online_interval",
        dest="rules_interval",
        type=int,
        default=5,
        help="Episodes per rule update batch (buffer→mine→codegen→verify→select).",
    )
    args = parser.parse_args()

    tasks = load_textcraft_tasks(args.task_file)
    num_games = len(tasks)
    seeds = [task["seed"] for task in tasks]

    apply_api_config()

    io_dir = args.io_dir.strip() or _default_io_dir()
    results_dir = Path(args.results_dir.strip() or (Path(io_dir) / "results"))
    traj_root = Path(args.traj_dir.strip() or (Path(io_dir) / "traj_data" / "textcraft" / "buffer_traj"))
    ensure_dir(results_dir)
    ensure_dir(traj_root)
    ensure_dir(Path(io_dir) / "symbolic_knowledge" / "textcraft")

    # LLM
    model_name = get_api_model("s1_trajectory", os.getenv("OPENAI_MODEL", "").strip() or "kimi-k2-0905")
    client = make_openai_client(model_name=model_name)
    llm_retry = RetryConfig(max_retries=11, base_delay_s=30.0, max_delay_s=30.0)

    llm = make_chat_llm(
        client,
        config=LLMConfig(
            model=model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retry=llm_retry,
        ),
        system_prompt=REACT_SYSTEM_PROMPT,
    )

    # Env
    TextCraft = get_textcraft_env()
    env = TextCraft()

    rules_engine: RulesEngine | None = None
    default_pruned_code_path = Path(io_dir) / "symbolic_knowledge" / "textcraft" / "pruned_rules_code.json"
    pruned_code_path = args.rules_code_path.strip() or str(default_pruned_code_path)

    if args.online_rules:
        rules_engine = RulesEngine()
        _console("[OnlineRules] Enabled: starting from empty rules.")

    buffer = None
    miner = None
    verifier = None
    if args.rules_interval <= 0:
        raise SystemExit("--rules_interval must be > 0")
    default_model = os.getenv("OPENAI_MODEL", "").strip() or model_name
    miner_model = get_api_model("rule_miner", default_model)
    verifier_model = get_api_model("rule_verifier", default_model)
    buffer = Buffer(io_dir=io_dir, env_name="textcraft")
    miner_client = make_openai_client(model_name=miner_model)
    verifier_client = make_openai_client(model_name=verifier_model)
    miner = RuleMiner(
        io_dir=io_dir,
        env_name="textcraft",
        client=miner_client,
        llm_config=LLMConfig(model=miner_model, temperature=0.0, max_tokens=2048, retry=llm_retry),
    )
    verifier = RuleVerifier(
        io_dir=io_dir,
        env_name="textcraft",
        client=verifier_client,
        llm_config=LLMConfig(model=verifier_model, temperature=0.0, max_tokens=2048, retry=llm_retry),
    )

    # Run
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = results_dir / f"s1_textcraft_collect_n{num_games}_steps{args.max_steps}_{now}.json"

    _console(
        f"[RunConfig] model={model_name} num_games={num_games} seeds={seeds[0]}..{seeds[-1]} "
        f"max_steps={args.max_steps} online_rules={bool(args.online_rules)} rules_interval={args.rules_interval}"
    )
    _console(f"[RunConfig] results_dir={results_dir}")
    _console(f"[RunConfig] traj_dir={traj_root}")

    results: dict[str, Any] = {}
    successes = 0
    for idx, seed in enumerate(seeds):
        ep_dir = traj_root / f"traj_{seed}"
        ensure_dir(ep_dir)
        # 跑一个task
        ep = run_episode(
            env,
            seed=seed,
            llm=llm,
            max_steps=args.max_steps,
            verbose=args.verbose,
            rules=rules_engine,
            max_rule_retries=args.max_rule_retries,
            save_dir=ep_dir,
            expected_task=tasks[idx],
        )
        results[f"env_{seed}"] = ep
        successes += int(bool(ep["success"]))

        # Rule mining/verification every interval episodes (offline by default; online additionally reloads for gating).
        if buffer is not None and miner is not None and verifier is not None:
            at_boundary = ((idx + 1) % int(args.rules_interval) == 0) or (idx + 1 == len(seeds))
            if at_boundary:
                current_interval = int(args.rules_interval)
                if idx + 1 == len(seeds) and (idx + 1) % int(args.rules_interval):
                    current_interval = (idx + 1) % int(args.rules_interval)
                start_task_id = seed - current_interval + 1

                try:
                    prefix = "[OnlineRules]" if args.online_rules else "[OfflineRules]"
                    _console(f"{prefix} Updating rules from batch {start_task_id} - {start_task_id + current_interval}")
                    # step1: transition 分类
                    _console(f"{prefix} Step 1/5: building transition buffers")
                    buffer.string_buffer_for_transitions_pure(current_interval, start_task_id, cleanup=False)
                    # step2: 提取规则
                    _console(f"{prefix} Step 2/5: mining rules")
                    miner.get_rules_all()
                    # step3: 规则代码化
                    _console(f"{prefix} Step 3/5: converting rules to code")
                    verifier.rules_code_all()
                    # step4: 规则验证
                    _console(f"{prefix} Step 4/5: verifying rule functions")
                    verifier.functions_verification()
                    # step5: 规则选择
                    _console(f"{prefix} Step 5/5: selecting usable rules")
                    verifier.select_rules()
                    _console(f"{prefix} Rule update finished.")

                    # If user provided a custom code path, mirror the latest pruned file there.
                    if args.rules_code_path.strip() and default_pruned_code_path.exists():
                        try:
                            Path(pruned_code_path).write_text(
                                default_pruned_code_path.read_text(encoding="utf-8"),
                                encoding="utf-8",
                            )
                        except Exception:
                            pass

                    # Online: reload pruned rules for subsequent episodes.
                    if args.online_rules and rules_engine is not None:
                        rules_engine.load_code_file(pruned_code_path)
                        _console(f"[OnlineRules] Reloaded {len(rules_engine.functions)} rule functions.")
                except Exception as e:
                    prefix = "[OnlineRules]" if args.online_rules else "[OfflineRules]"
                    _console(f"{prefix}[Error] {type(e).__name__}: {e}")

        rate = successes / max(1, idx + 1)
        results["overall"] = {
            "success_rate": rate,
            "num_games": num_games,
            "seeds": seeds,
            "max_steps": args.max_steps,
            "model": model_name,
            "temperature": args.temperature,
        }
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _console(f"Saved results to: {out_path}")
    _console(f"Success rate: {rate:.3f} ({successes}/{num_games})")


if __name__ == "__main__":
    main()
