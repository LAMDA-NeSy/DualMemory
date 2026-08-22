from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from api_config import apply_api_config, get_api_model
from llm_client import LLMConfig, make_chat_llm, make_openai_client
from llm_metrics import get_metrics, reset_metrics, write_metrics
from progress_memory import ProgressMemoryPlanner
from parsing import (
    infer_action_success,
    is_craft_action,
    is_get_action,
    is_think,
    normalize_item_name,
    parse_action,
    parse_goal,
    parse_inventory_observation,
    parse_recipes,
    to_action_json,
)
from prompts import REACT_SYSTEM_PROMPT, append_react_history, build_react_user_prompt
from rules_engine import RulesEngine
from textcraft_env import get_textcraft_env
from task_file import load_textcraft_tasks, validate_textcraft_task
from utils import ensure_dir, write_json, write_text


def _default_io_dir() -> str:
    return str(Path(__file__).resolve().parent)


def _console(msg: str = "") -> None:
    print(msg, flush=True)


def _format_inventory(inv: dict[str, int]) -> str:
    if not inv:
        return "Inventory: (empty)"
    parts = [f"[{k}] ({int(v)})" for k, v in sorted(inv.items())]
    return "Inventory: " + " ".join(parts)


_WOOD_VARIANTS = (
    "oak",
    "spruce",
    "birch",
    "jungle",
    "acacia",
    "dark oak",
    "crimson",
    "warped",
)
_STONE_CRAFTING_MATERIALS = {
    "stone",
    "cobblestone",
    "blackstone",
    "cobbled deepslate",
}


def _item_key(name: str) -> str:
    return normalize_item_name(str(name or "")).lower()


def _item_aliases(name: str) -> set[str]:
    item = _item_key(name)
    if not item:
        return set()

    aliases = {item}
    if item.endswith(" planks"):
        aliases.add("planks")
    if item.endswith(" logs") or item.endswith(" log") or item.endswith(" stems") or item.endswith(" stem"):
        aliases.add("logs")
    if any(item in {f"{wood} slab", f"{wood} slabs"} for wood in _WOOD_VARIANTS):
        aliases.add("wooden slabs")
    if item in _STONE_CRAFTING_MATERIALS:
        aliases.add("stone crafting materials")
    return aliases


def _build_effective_inventory(raw_inventory: dict[str, int]) -> dict[str, int]:
    effective: dict[str, int] = {}
    for item, count in raw_inventory.items():
        key = _item_key(item)
        qty = int(count or 0)
        if not key or qty <= 0:
            continue
        effective[key] = int(effective.get(key, 0)) + qty
        for alias in _item_aliases(key):
            if alias == key:
                continue
            effective[alias] = int(effective.get(alias, 0)) + qty
    return effective


def _consume_inventory_item(raw_inventory: dict[str, int], item: str, count: int) -> bool:
    remaining = int(count or 0)
    if remaining <= 0:
        return True

    need = _item_key(item)
    if not need:
        return False

    candidates: list[str] = []
    if need in raw_inventory:
        candidates.append(need)
    for key in sorted(raw_inventory):
        if key == need:
            continue
        if need in _item_aliases(key):
            candidates.append(key)

    for key in candidates:
        available = int(raw_inventory.get(key, 0))
        if available <= 0:
            continue
        used = min(available, remaining)
        raw_inventory[key] = available - used
        if raw_inventory[key] <= 0:
            raw_inventory.pop(key, None)
        remaining -= used
        if remaining <= 0:
            return True
    return False


def _format_fewshot_segment(segment: list[dict[str, str]]) -> str:
    if not segment:
        return ""
    lines: list[str] = []
    for step in segment:
        a = str(step.get("action") or "").strip()
        o = str(step.get("observation") or "").strip()
        if a:
            lines.append(f"> {a}")
        if o:
            lines.append(o)
    return "\n".join(lines).strip()


def _format_recipe_lines(recipes: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for recipe in recipes or []:
        if not isinstance(recipe, dict):
            continue
        raw = str(recipe.get("raw") or "").strip()
        if raw:
            lines.append(raw)
            continue
        output = recipe.get("output") or {}
        out_item = str(output.get("item") or "").strip()
        out_count = int(output.get("count") or 0)
        inputs = recipe.get("inputs") or []
        input_parts: list[str] = []
        for ing in inputs:
            if not isinstance(ing, dict):
                continue
            ing_item = str(ing.get("item") or "").strip()
            ing_count = ing.get("count")
            if ing_item and isinstance(ing_count, int):
                input_parts.append(f"{ing_count} {ing_item}")
        if out_item and out_count > 0 and input_parts:
            lines.append(f"craft {out_count} {out_item} using {', '.join(input_parts)}")
    return "\n".join(lines).strip()


def _format_fewshot_segments(hits: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for i, hit in enumerate(hits or [], start=1):
        seg = hit.get("core_segment") or hit.get("segment") or []
        if not isinstance(seg, list) or not seg:
            continue
        milestone_text = str(hit.get("milestone") or "").strip()
        recipes_text = _format_recipe_lines(list(hit.get("recipes") or []))

        header = f"Demonstration {i}"
        if milestone_text:
            header += f" (Retrieved Milestone: {milestone_text})"
        parts: list[str] = [header + ":"]
        if recipes_text:
            parts.append("Crafting commands:")
            parts.append(recipes_text)
        parts.append("Trajectory:")
        parts.append(_format_fewshot_segment(seg))
        blocks.append("\n".join(parts).strip())
    return "\n\n".join(blocks).strip()


def _format_milestone_guide(guide: list[str]) -> str:
    return "\n".join(f"{i}. {m}" for i, m in enumerate(guide or [], start=1) if str(m or "").strip())


def _compact_fewshot_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for h in hits or []:
        if not isinstance(h, dict):
            continue
        out.append(
            {
                "task_id": h.get("task_id"),
                "milestone_id": h.get("milestone_id"),
                "order": h.get("order"),
                "milestone": h.get("milestone"),
                "milestone_item": h.get("milestone_item"),
                "milestone_count": h.get("milestone_count"),
                "recipes": h.get("recipes"),
                "action_indices": h.get("action_indices"),
                "core_segment": h.get("core_segment"),
                "segment": h.get("segment"),
            }
        )
    return out


def _build_progress_memory_prompt(
    *,
    problem: str,
    history: str,
    milestone_guide: list[str],
    current_milestone_idx: int,
    fewshot: str,
) -> str:
    guide_text = _format_milestone_guide(milestone_guide) if milestone_guide else ""

    current_text = ""
    if milestone_guide and 0 <= current_milestone_idx < len(milestone_guide):
        current_text = f"{current_milestone_idx+1}. {milestone_guide[current_milestone_idx]}"

    # blocks: list[str] = [problem.strip(), "", f"[State]\n{inv_line}"]
    blocks: list[str] = [problem.strip(), ""]
    if guide_text:
        blocks.append(f"\n[Milestone Guide]\n{guide_text}")
    if current_text:
        blocks.append(f"\n[Current Milestone]\n{current_text}")
    if fewshot:
        blocks.append(f"\n[Few-shot Example]\n{fewshot}")
    if history.strip():
        blocks.append(f"\n[Current Trajectory]\n{history.strip()}")

    return "\n".join(blocks).rstrip() + "\n> "


def run_episode(
    env,
    *,
    seed: int,
    llm,  # action model
    planner: ProgressMemoryPlanner | None,
    progress_memory: bool,
    max_steps: int,
    verbose: bool,
    rules: RulesEngine | None,
    max_rule_retries: int,
    top_milestones: int,
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

    history = ""
    log_trajectory = f"Action: reset\nObservation: {problem}\n\n"
    steps: list[dict[str, Any]] = []

    inventory: dict[str, int] = {}
    imagination_events: list[dict[str, Any]] = []

    llm_calls = 0  # total (action + PROGRESS_MEMORY planner/judge)
    max_patience = 8
    patience_ctr = 0

    goal = parse_goal(problem) or {"item": "", "count": 1}
    recipes = parse_recipes(problem)
    craftable_items = sorted({r.get("output", {}).get("item") for r in recipes if isinstance(r, dict) and isinstance(r.get("output", {}).get("item"), str)})

    milestone_guide: list[str] = []
    if progress_memory and planner is not None:
        before = int(getattr(planner, "llm_calls", 0))
        # 生成milestone guide
        milestone_guide = planner.build_milestone_guide(problem)
        llm_calls += int(getattr(planner, "llm_calls", 0)) - before
        if milestone_guide:
            _console(f"[PROGRESS_MEMORY] Seed={seed} milestones={len(milestone_guide)}")
            for i, m in enumerate(milestone_guide, start=1):
                _console(f"  {i}. {m}")
            log_trajectory += (
                "Action: [PROGRESS_MEMORY] milestone_guide\n"
                f"Observation: {_format_milestone_guide(milestone_guide)}\n\n"
            )
        else:
            log_trajectory += "Action: [PROGRESS_MEMORY] milestone_guide\nObservation: (empty)\n\n"
    current_milestone_idx = 0
    logged_fewshot_for: set[int] = set()

    # 把action和observation记录下来，并更新inventory
    def _append_step(action: str, observation: str, reward: float, done: bool) -> None:
        nonlocal history, log_trajectory, inventory
        action = str(action).strip()
        observation = str(observation).strip()
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

        if progress_memory:
            history += f"> {action}\n{observation}\n\n"
        else:
            history = append_react_history(history, action, observation)
        log_trajectory += f"Action: {action}\nObservation: {observation}\n\n"

        action_json = to_action_json(action) or {}
        if action_json.get("name") == "inventory":
            inv = parse_inventory_observation(observation)
            if inv is not None:
                inventory = {_item_key(k): int(v) for k, v in dict(inv).items() if _item_key(k)}
        elif action_success and is_get_action(action):
            args = action_json.get("args") or {}
            item = args.get("item")
            count = args.get("count")
            if isinstance(item, str) and isinstance(count, int):
                key = _item_key(item)
                inventory[key] = int(inventory.get(key, 0)) + int(count)
        elif action_success and is_craft_action(action):
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
                        _consume_inventory_item(inventory, ing_item, ing_count)

            if isinstance(out_item, str) and isinstance(out_count, int):
                out_key = _item_key(out_item)
                inventory[out_key] = int(inventory.get(out_key, 0)) + int(out_count)

    def _append_imagination_event(
        *,
        step_idx: int,
        attempt: int,
        candidate_action: str,
        prompt_observation: str,
        reason: str,
        rule_id: str = "",
        feedback: str = "",
        suggestion: str = "",
        action_json: dict[str, Any] | None = None,
        log_observation: str | None = None,
    ) -> None:
        nonlocal history, log_trajectory, imagination_events
        cand = str(candidate_action or "").strip()
        prompt_obs = str(prompt_observation or "").strip()
        log_obs = prompt_obs if log_observation is None else str(log_observation or "").strip()

        imagination_events.append(
            {
                "step_idx": int(step_idx),
                "attempt": int(attempt),
                "candidate_action": cand,
                "reason": str(reason or "").strip(),
                "rule_id": str(rule_id or "").strip(),
                "feedback": str(feedback or "").strip(),
                "suggestion": str(suggestion or "").strip(),
                "action_json": action_json,
            }
        )

        # Also write into prompt-visible history and episode.log for debugging.
        action_line = f"Action in Imagination (attempt {int(attempt)}): {cand}".strip()
        if progress_memory:
            history += f"> {action_line}\n{prompt_obs}\n\n"
        else:
            history = append_react_history(history, action_line, prompt_obs)
        log_trajectory += f"Action: {action_line}\nObservation: {log_obs}\n\n"
        _console(f"Action: {action_line}")
        _console(f"Observation: {log_obs}\n")



    reward = 0.0
    done = False
    terminate_reason = "max_steps"

    for _step_idx in range(1, max_steps + 1):
        current_milestone_text = ""
        progress_check: dict[str, Any] | None = None
        if progress_memory and planner is not None and milestone_guide:
            # 提取干净轨迹（剔除think）
            recent_traj = [
                {"action": str(s.get("action") or "").strip(), "observation": str(s.get("observation") or "").strip()}
                for s in steps
                if isinstance(s, dict) and not is_think(str(s.get("action") or ""))
            ]
            before = int(getattr(planner, "llm_calls", 0))
            # 确定当前milestone
            current_milestone_idx = planner.determine_current_milestone_idx(
                task_text=problem,
                milestone_guide=milestone_guide,
                current_milestone_idx=current_milestone_idx,
                recent_trajectory=recent_traj,
                inventory_line=_format_inventory(_build_effective_inventory(inventory)),
            )
            llm_calls += int(getattr(planner, "llm_calls", 0)) - before
            progress_check = dict(getattr(planner, "last_progress_check", {}) or {})
            current_milestone_text = str(milestone_guide[current_milestone_idx] or "").strip()
            log_trajectory += (
                f"Action: [PROGRESS_MEMORY] current_milestone {current_milestone_idx + 1}/{len(milestone_guide)}\n"
                f"Observation: {current_milestone_text}\n\n"
            )

        fewshot = ""
        fewshot_hits: list[dict[str, Any]] = []
        if (
            progress_memory
            and planner is not None
            and milestone_guide
            and 0 <= current_milestone_idx < len(milestone_guide)
            and int(top_milestones or 0) > 0
            and planner.has_library()
        ):
            current_milestone = str(milestone_guide[current_milestone_idx] or "").strip()
            hit = planner.retrieve_milestone_hits(
                current_milestone,
                top_k=int(top_milestones or 1),
                exclude_task_ids=None,
            )
            if hit:
                fewshot_hits = _compact_fewshot_hits(hit)
                fewshot = _format_fewshot_segments(hit)
                if current_milestone_idx not in logged_fewshot_for:
                    log_trajectory += f"Action: [PROGRESS_MEMORY] fewshot_text\nObservation: {fewshot}\n\n"
                    logged_fewshot_for.add(current_milestone_idx)

        if progress_memory:
            prompt = _build_progress_memory_prompt(
                problem=problem,
                history=history,
                milestone_guide=milestone_guide,
                current_milestone_idx=current_milestone_idx,
                fewshot=fewshot,
            )
        else:
            prompt = build_react_user_prompt(problem, history)

        candidate_raw = llm(prompt, stop=["\n"])
        llm_calls += 1
        candidate_action = parse_action(candidate_raw) or "inventory"

        imagination_start = len(imagination_events)
        if rules is not None and rules.functions:
            retries_left = max(0, int(max_rule_retries))
            imagination_attempt = 0
            last_action = candidate_action

            while retries_left > 0:
                # think直接跳过
                if is_think(candidate_action) or candidate_action == "inventory":
                    break
                # 语法不对直接拦截
                action_json = to_action_json(candidate_action)
                if not action_json or action_json.get("name") not in {"get", "craft"}:
                    imagination_attempt += 1
                    _append_imagination_event(
                        step_idx=_step_idx,
                        attempt=imagination_attempt,
                        candidate_action=candidate_action,
                        prompt_observation="Invalid action format. Output one valid action.",
                        reason="invalid_action_format",
                        action_json=action_json,
                    )
                    retries_left -= 1
                    if progress_memory:
                        prompt = _build_progress_memory_prompt(
                            problem=problem,
                            history=history,
                            milestone_guide=milestone_guide,
                            current_milestone_idx=current_milestone_idx,
                            fewshot=fewshot,
                        )
                    else:
                        prompt = build_react_user_prompt(problem, history)
                    candidate_raw = llm(prompt, stop=["\n"])
                    llm_calls += 1
                    candidate_action = parse_action(candidate_raw) or "inventory"
                    continue
                
                # 过一遍rule codes
                state = {
                    "goal": goal,
                    "recipes": recipes,
                    "craftable_items": craftable_items,
                    "inventory": _build_effective_inventory(inventory),
                    "inventory_known": True,
                }
                res = rules.check(state=state, action=action_json)
                if res.success:
                    break

                imagination_attempt += 1
                last_action = candidate_action
                msg = (res.feedback + " " + res.suggestion).strip()
                _append_imagination_event(
                    step_idx=_step_idx,
                    attempt=imagination_attempt,
                    candidate_action=candidate_action,
                    prompt_observation=msg,
                    log_observation=(f"[{res.rule_id}] {msg}".strip() if res.rule_id else msg),
                    reason="rule_blocked",
                    rule_id=res.rule_id,
                    feedback=res.feedback,
                    suggestion=res.suggestion,
                    action_json=action_json,
                )
                retries_left -= 1

                if progress_memory:
                    prompt = _build_progress_memory_prompt(
                        problem=problem,
                        history=history,
                        milestone_guide=milestone_guide,
                        current_milestone_idx=current_milestone_idx,
                        fewshot=fewshot,
                    )
                else:
                    prompt = build_react_user_prompt(problem, history)
                candidate_raw = llm(prompt, stop=["\n"])
                llm_calls += 1
                candidate_action = parse_action(candidate_raw) or "inventory"

            if retries_left <= 0:
                candidate_action = last_action

        # 确定动作了
        action = candidate_action

        if not progress_memory and "task completed" in action.lower():
            reward = 1.0
            done = True
            terminate_reason = "task_completed_thought"
            _console(f"Action: {action}\n")
            break
        if not progress_memory and "task failed" in action.lower():
            terminate_reason = "task_failed_thought"
            _console(f"Action: {action}\n")
            break

        if is_think(action):
            observation = "OK."
            step_reward = 0.0
            step_done = False
        else:
            try:
                observation, step_reward, step_done, _truncated, _info = env.step(action)
            except Exception as e:
                observation = f"Invalid action: {e}"
                step_reward = 0.0
                step_done = False

        _append_step(action, str(observation), float(step_reward or 0.0), bool(step_done))
        step_obj = steps[-1]
        if progress_memory:
            step_obj["milestone_guide"] = list(milestone_guide or [])
            step_obj["milestone_idx"] = int(current_milestone_idx)
            step_obj["milestone_total"] = int(len(milestone_guide or []))
            step_obj["current_milestone"] = current_milestone_text
            step_obj["progress_memory_progress_check"] = progress_check
            step_obj["fewshot_hits"] = fewshot_hits
            step_obj["fewshot_text"] = fewshot
        step_obj["imagination_events"] = imagination_events[imagination_start:]

        reward = float(step_reward or 0.0)
        done = bool(step_done)

        _console(f"Action: {action}")
        _console(f"Observation: {str(observation).strip()}\n")

        if not progress_memory and not is_think(action):
            if str(observation).startswith("Could not") or str(observation).strip() == "OK.":
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

    ensure_dir(save_dir)
    write_text(save_dir / "episode.log", log_trajectory.rstrip() + "\n")
    write_json(
        save_dir / "episode.json",
        {
            "seed": seed,
            "problem": problem,
            "success": bool(reward > 0),
            "reward": reward,
            "done": done,
            "steps": steps,
            "imagination_events": imagination_events,
            "llm_calls": llm_calls,
            "terminate_reason": terminate_reason,
            "milestone_guide": list(milestone_guide or []),
            "feasibility_memory": bool(rules is not None and rules.functions),
            "progress_memory": bool(progress_memory),
            "top_milestones": int(top_milestones),
            "max_rule_retries": int(max_rule_retries),
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
        "imagination_events": len(imagination_events),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 3: run TextCraft agent with (rules + milestones).")
    p.add_argument("--task_file", type=str, required=True, help="Published ordered TextCraft task JSON.")
    p.add_argument("--max_steps", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--io_dir", type=str, default="")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--traj_dir", type=str, default="")
    p.add_argument("--verbose", action="store_true")

    p.add_argument(
        "--progress_memory",
        dest="progress_memory",
        action="store_true",
        help="Enable PROGRESS_MEMORY: generate milestones with an LLM, retrieve milestone demos, and use an LLM judge to advance milestones.",
    )
    p.add_argument("--progress_memory_library", type=str, default="progress_memory_library.json")
    p.add_argument("--top_milestones", type=int, default=0)
    p.add_argument("--top_tasks", type=int, default=3)
    p.add_argument("--planner_model", type=str, default="")
    p.add_argument("--judge_model", type=str, default="")
    p.add_argument("--embedding_model", type=str, default=get_api_model("embedding", "all-mpnet-base-v2"))

    p.add_argument("--feasibility_memory", action="store_true")
    p.add_argument("--rules_code_path", type=str, default="")
    p.add_argument("--max_rule_retries", type=int, default=5)

    args = p.parse_args()

    tasks = load_textcraft_tasks(args.task_file)
    num_games = len(tasks)
    seeds = [task["seed"] for task in tasks]

    reset_metrics()
    apply_api_config()

    io_dir = args.io_dir.strip() or _default_io_dir()
    results_dir = Path(args.results_dir.strip() or (Path(io_dir) / "results"))
    traj_root = Path(args.traj_dir.strip() or (Path(io_dir) / "traj_data" / "textcraft" / "inference_traj"))
    ensure_dir(results_dir)
    ensure_dir(traj_root)

    model_name = get_api_model("s3_action", os.getenv("OPENAI_MODEL", "").strip() or "kimi-k2-0905")
    client = make_openai_client(model_name=model_name)
    llm = make_chat_llm(
        client,
        config=LLMConfig(
            model=model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        system_prompt=REACT_SYSTEM_PROMPT,
        default_stop=["\n"],
    )
    TextCraft = get_textcraft_env()
    env = TextCraft()

    rules_engine: RulesEngine | None = None
    if args.feasibility_memory:
        rules_engine = RulesEngine()
        code_path = args.rules_code_path.strip() or str(Path(io_dir) / "symbolic_knowledge" / "textcraft" / "pruned_rules_code.json")
        if not Path(code_path).is_file():
            raise FileNotFoundError(f"--feasibility_memory requires Stage 1 rules: {code_path}")
        rules_engine.load_code_file(code_path)
        print(f"[S3] Loaded {len(rules_engine.functions)} rule functions from {code_path}")

    planner: ProgressMemoryPlanner | None = None
    if args.progress_memory:
        lib_path = args.progress_memory_library.strip()
        if not lib_path:
            raise ValueError("--progress_memory requires --progress_memory_library produced by Stage 2.")
        full = str(Path(io_dir) / lib_path) if lib_path and not Path(lib_path).is_absolute() else lib_path
        if not Path(full).is_file():
            raise FileNotFoundError(f"Progress memory library not found: {full}")
        planner_model = args.planner_model.strip() or get_api_model("progress_memory_planner", model_name)
        judge_model = args.judge_model.strip() or get_api_model("progress_memory_judge", planner_model)
        planner = ProgressMemoryPlanner(
            library_path=full,
            prompt_dir=str(Path(io_dir) / "prompts"),
            planner_model=planner_model,
            judge_model=judge_model,
            embedding_model=args.embedding_model,
            top_tasks=args.top_tasks,
            top_milestones=args.top_milestones,
        )
        if lib_path:
            if planner.has_library():
                print(f"[S3] Loaded progress memory library: {full} (tasks={len(planner.library.tasks)})")
            else:
                print(f"[S3] No progress memory library data at: {full} (PROGRESS_MEMORY will run without retrieval demos)")

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = results_dir / f"s3_textcraft_our_design_n{num_games}_steps{args.max_steps}_{now}.json"

    results: dict[str, Any] = {}
    successes = 0
    for idx, seed in enumerate(seeds):
        ep_dir = traj_root / f"traj_{seed}"
        ensure_dir(ep_dir)
        ep = run_episode(
            env,
            seed=seed,
            llm=llm,
            planner=planner,
            progress_memory=bool(args.progress_memory),
            max_steps=args.max_steps,
            verbose=args.verbose,
            rules=rules_engine,
            max_rule_retries=args.max_rule_retries,
            top_milestones=args.top_milestones,
            save_dir=ep_dir,
            expected_task=tasks[idx],
        )
        results[f"env_{seed}"] = ep
        successes += int(bool(ep["success"]))

        rate = successes / max(1, idx + 1)
        metrics = get_metrics()
        results["overall"] = {
            "success_rate": rate,
            "num_games": num_games,
            "seeds": seeds,
            "max_steps": args.max_steps,
            "model": model_name,
            "temperature": args.temperature,
            "effective_rules": bool(args.feasibility_memory),
            "llm_metrics": {
                k: metrics[k]
                for k in (
                    "llm_call_count",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "latency_seconds",
                    "avg_latency_seconds",
                )
            },
        }
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics_path = results_dir / "llm_metrics.json"
    write_metrics(
        metrics_path,
        extra={
            "results_path": str(out_path),
            "num_games": num_games,
            "seeds": seeds,
            "model": model_name,
        },
    )
    print(f"Saved results to: {out_path}")
    print(f"Saved LLM metrics to: {metrics_path}")
    print(f"Success rate: {rate:.3f} ({successes}/{num_games})")


if __name__ == "__main__":
    main()
