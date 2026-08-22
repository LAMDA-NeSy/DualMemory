from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from parsing import (
    infer_action_success,
    parse_goal,
    parse_inventory_observation,
    parse_recipes,
    to_action_json,
)
from utils import ensure_dir, list_traj_ids, read_json, write_json


@dataclass
class EpisodeStep:
    action: str
    observation: str
    reward: float | None = None
    done: bool | None = None
    action_success: bool | None = None


@dataclass
class Episode:
    task_id: int
    seed: int
    problem: str
    steps: list[EpisodeStep]
    success: bool = False
    reward: float = 0.0


def _parse_react_log_trajectory(text: str) -> tuple[str, list[EpisodeStep]]:
    """
    Parse the text trajectory log format:
      Action: ...
      Observation: ...

    Returns: (problem_text, steps)
    """
    lines = (text or "").splitlines()
    steps: list[EpisodeStep] = []
    cur_action: str | None = None
    cur_obs_lines: list[str] = []

    # Some trajectory logs include "Action: reset" followed by the problem.
    for line in lines:
        if line.startswith("Action: "):
            if cur_action is not None:
                steps.append(EpisodeStep(action=cur_action, observation="\n".join(cur_obs_lines).strip()))
            cur_action = line[len("Action: ") :].strip()
            cur_obs_lines = []
            continue
        if line.startswith("Observation: "):
            cur_obs_lines.append(line[len("Observation: ") :])
            continue
        if cur_action is not None:
            # Multi-line observation continuation.
            cur_obs_lines.append(line)
    if cur_action is not None:
        steps.append(EpisodeStep(action=cur_action, observation="\n".join(cur_obs_lines).strip()))

    problem = ""
    if steps and steps[0].action == "reset":
        problem = steps[0].observation.strip()
        steps = steps[1:]
    return problem, steps


def _load_episode_from_file(path: Path, *, task_id: int) -> Episode:
    data = json.loads(path.read_text(encoding="utf-8"))
    problem = str(data.get("problem") or "")
    seed = int(data.get("seed", task_id))
    steps_raw = data.get("steps") or []
    steps: list[EpisodeStep] = []
    for item in steps_raw:
        if not isinstance(item, dict):
            continue
        steps.append(
            EpisodeStep(
                action=str(item.get("action") or ""),
                observation=str(item.get("observation") or ""),
                reward=float(item.get("reward")) if item.get("reward") is not None else None,
                done=bool(item.get("done")) if item.get("done") is not None else None,
                action_success=bool(item.get("action_success")) if item.get("action_success") is not None else None,
            )
        )
    return Episode(
        task_id=task_id,
        seed=seed,
        problem=problem,
        steps=steps,
        success=bool(data.get("success", False)),
        reward=float(data.get("reward", 0.0) or 0.0),
    )


class Buffer:
    """
    Convert TextCraft trajectories into (state, action, action_result) transitions.

    - correct: action_success == True
    - wrong:   action_success == False
    """

    def __init__(self, *, io_dir: str, env_name: str = "textcraft") -> None:
        self.io_dir = io_dir
        self.env_name = env_name
        self.traj_dir = os.path.join(io_dir, "traj_data", env_name, "buffer_traj")
        self.fact_dir = os.path.join(io_dir, "traj_data", env_name)

        self.record_wrong: dict[str, list[dict[str, Any]]] = {}
        self.record_correct: dict[str, list[dict[str, Any]]] = {}

    def _episode_paths(self, traj_id: int) -> tuple[Path | None, Path | None]:
        base = Path(self.traj_dir) / f"traj_{traj_id}"
        if not base.exists():
            return None, None
        json_path = base / "episode.json"
        log_path = base / "episode.log"
        return (json_path if json_path.exists() else None), (log_path if log_path.exists() else None)

    def _load_episode(self, traj_id: int) -> Episode:
        json_path, log_path = self._episode_paths(traj_id)
        if json_path is not None:
            return _load_episode_from_file(json_path, task_id=traj_id)
        if log_path is None:
            raise FileNotFoundError(f"Missing episode.json/episode.log for traj_{traj_id} under {self.traj_dir}")
        problem, steps = _parse_react_log_trajectory(log_path.read_text(encoding="utf-8"))
        success = any((s.observation or "").startswith("Crafted ") for s in steps)  # weak fallback
        return Episode(task_id=traj_id, seed=traj_id, problem=problem, steps=steps, success=success, reward=1.0 if success else 0.0)

    def _build_transitions(self, episode: Episode) -> list[dict[str, Any]]:
        goal = parse_goal(episode.problem) or {"item": "", "count": 1}
        recipes = parse_recipes(episode.problem)
        craftable_items = sorted({r["output"]["item"] for r in recipes if isinstance(r, dict) and "output" in r and r["output"].get("item")})

        inventory: dict[str, int] = {}
        inventory_known = False

        transitions: list[dict[str, Any]] = []

        for step_idx, step in enumerate(episode.steps):
            action = (step.action or "").strip()
            obs = (step.observation or "").strip()

            # Update action_success if not present.
            action_success = step.action_success
            if action_success is None:
                action_success = infer_action_success(action, obs)

            # Build initial_state before applying this step.
            state = {
                "goal": goal,
                "recipes": recipes,
                "craftable_items": craftable_items,
                "inventory": dict(inventory) if inventory_known else {},
                "inventory_known": bool(inventory_known),
            }

            action_json = to_action_json(action)

            # Only record environment-interacting actions.
            if action_json and action_json.get("name") in {"get", "craft"}:
                transitions.append(
                    {
                        "task_id": episode.task_id,
                        "seed": episode.seed,
                        "step": step_idx,
                        "initial_state": state,
                        "action": action_json,
                        "action_result": bool(action_success),
                        "observation": obs,
                    }
                )

            # Apply state update.
            if action_json:
                name = action_json.get("name")
                if name == "inventory":
                    inv = parse_inventory_observation(obs)
                    if inv is not None:
                        inventory = dict(inv)
                        inventory_known = True
                elif inventory_known and bool(action_success) and name == "get":
                    args = action_json.get("args") or {}
                    item = args.get("item")
                    count = args.get("count")
                    if isinstance(item, str) and isinstance(count, int):
                        inventory[item] = int(inventory.get(item, 0)) + int(count)
                elif inventory_known and bool(action_success) and name == "craft":
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

        return transitions

    def string_buffer_for_transitions_pure(self, interval: int, task_id: int, *, cleanup: bool = False) -> None:
        record_wrong_temp: dict[str, list[dict[str, Any]]] = {}
        record_correct_temp: dict[str, list[dict[str, Any]]] = {}

        # 遍历task
        for kk in range(interval):
            traj_id = task_id + kk
            episode = self._load_episode(traj_id)
            transitions = self._build_transitions(episode)
            # 遍历transition
            for t in transitions:
                act_name = str((t.get("action") or {}).get("name") or "")
                if not act_name:
                    continue
                # 根据action result分类
                if t.get("action_result") is True:
                    self.record_correct.setdefault(act_name, []).append(t)
                    record_correct_temp.setdefault(act_name, []).append(t)
                else:
                    self.record_wrong.setdefault(act_name, []).append(t)
                    record_wrong_temp.setdefault(act_name, []).append(t)

            if cleanup:
                # Mirror ALFWorld cleanup option: delete processed traj dirs.
                base = Path(self.traj_dir) / f"traj_{traj_id}"
                if base.exists():
                    for child in base.iterdir():
                        try:
                            child.unlink()
                        except Exception:
                            pass
                    try:
                        base.rmdir()
                    except Exception:
                        pass

        ensure_dir(self.fact_dir)
        write_json(os.path.join(self.fact_dir, "buffer_wrong_all.json"), self.record_wrong, indent=2)
        write_json(os.path.join(self.fact_dir, "buffer_correct_all.json"), self.record_correct, indent=2)
        write_json(os.path.join(self.fact_dir, "buffer_wrong_temp.json"), record_wrong_temp, indent=2)
        write_json(os.path.join(self.fact_dir, "buffer_correct_temp.json"), record_correct_temp, indent=2)

    def build_all(self, *, cleanup: bool = False) -> None:
        traj_ids = list_traj_ids(self.traj_dir)
        if not traj_ids:
            raise RuntimeError(f"No trajectories found under {self.traj_dir}")
        for traj_id in traj_ids:
            self.string_buffer_for_transitions_pure(1, traj_id, cleanup=cleanup)

    def import_from_react_results(self, results_json_path: str) -> None:
        """
        Convenience: convert a trajectory results JSON into traj_data/buffer_traj.

        This is useful when you already have trajectory logs and want to mine rules/milestones offline.
        """
        data = read_json(results_json_path)
        out_root = Path(self.traj_dir)
        ensure_dir(out_root)

        for key, item in data.items():
            if not isinstance(key, str) or not key.startswith("env_"):
                continue
            try:
                seed = int(key.split("env_", 1)[1])
            except Exception:
                continue
            traj_text = str(item.get("trajectory") or "")
            problem, steps = _parse_react_log_trajectory(traj_text)
            ep = {
                "seed": seed,
                "task_id": seed,
                "problem": problem,
                "success": bool(item.get("success", False)),
                "reward": float(item.get("reward", 0.0) or 0.0),
                "steps": [
                    {
                        "action": s.action,
                        "observation": s.observation,
                        "reward": s.reward,
                        "done": s.done,
                        "action_success": infer_action_success(s.action, s.observation),
                    }
                    for s in steps
                ],
            }
            traj_dir = out_root / f"traj_{seed}"
            ensure_dir(traj_dir)
            (traj_dir / "episode.json").write_text(json.dumps(ep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
