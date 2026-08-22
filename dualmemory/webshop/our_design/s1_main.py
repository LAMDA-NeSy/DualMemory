import argparse
import json
import os
from typing import Any, Dict, List

from api_config import get_api_model
from s1_webshop_trial import run_trial
from task_file import load_webshop_tasks


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--run_name", type=str, default="s1_webshop_run")
    parser.add_argument("--task_file", type=str, required=True, help="Published ordered WebShop task JSON.")
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--defer_rules", action="store_true")
    parser.add_argument("--online_rules", action="store_true")
    parser.add_argument(
        "--model",
        type=str,
        default=get_api_model("s1_trajectory", "gpt-4o-mini"),
        help="LLM model for trajectory collection",
    )
    parser.add_argument(
        "--base_prompt_file",
        type=str,
        default=os.path.join(os.path.abspath(os.path.dirname(__file__)), "prompts", "s1_react_prompt1.txt"),
        help="Optional ReAct-style init prompt file.",
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

    tasks = load_webshop_tasks(args.task_file)

    logging_dir = _resolve_run_dir(args.run_name)
    os.makedirs(logging_dir, exist_ok=True)
    world_log_path = os.path.join(logging_dir, "world.log")

    base_prompt = ""
    with open(args.base_prompt_file, "r", encoding="utf-8") as f:
        base_prompt = f.read().rstrip() + "\n"

    env_configs: List[Dict[str, Any]] = []
    for task in tasks:
        env_configs.append({"task_id": task["task_id"], "is_success": False})

    for trial_idx in range(args.num_trials):
        trial_log_path = os.path.join(logging_dir, f"trial_{trial_idx}.log")
        if os.path.exists(trial_log_path):
            open(trial_log_path, "w").close()

        run_trial(
            trial_log_path=trial_log_path,
            world_log_path=world_log_path,
            trial_idx=trial_idx,
            env_configs=env_configs,
            model=args.model,
            defer_rules=args.defer_rules,
            online_rules=args.online_rules,
            interval=args.interval,
            max_steps=args.max_steps,
            base_prompt=base_prompt,
            task_texts={task["task_id"]: task["task_text"] for task in tasks},
            session_ids={task["task_id"]: task["session_idx"] for task in tasks},
        )

        with open(os.path.join(logging_dir, f"env_results_trial_{trial_idx}.json"), "w", encoding="utf-8") as f:
            json.dump(env_configs, f, indent=2)


if __name__ == "__main__":
    main()
