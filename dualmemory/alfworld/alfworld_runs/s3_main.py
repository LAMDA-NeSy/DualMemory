import os
import json
import argparse

from s3_alfworld_trial import run_trial
from api_config import get_api_model

import re
from typing import Any, List, Dict, Optional, Tuple

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, help="The number of trials to run")
    parser.add_argument("--num_envs", type=int, help="The number of environments per trial")
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--is_resume", action='store_true', help="To resume run")
    parser.add_argument("--resume_dir", type=str, help="If resume, the logging directory", default="")
    parser.add_argument("--start_trial_num", type=int, help="If resume, the start trial num", default=0)
    parser.add_argument(
        "--model",
        type=str,
        default=get_api_model("s3_trajectory", "deepseek-ai/DeepSeek-V3"),
        help="The model to use. One of `gpt-4`, `gpt-3.5-turbo`, or `text-davinci-003",
    )
    parser.add_argument("--progress_memory", action='store_true', help="Enable Progress Memory guidance (also enables local milestone-level few-shot demos)")
    parser.add_argument("--progress_memory_library", type=str, default="", help="Path to Progress Memory library JSON")
    parser.add_argument("--feasibility_memory", action='store_true', help="Enable Feasibility Memory rule checking")
    parser.add_argument(
        "--task_file",
        type=str,
        required=True,
        help="Ordered ALFWorld task JSON file with relative gamefile paths",
    )
    parser.add_argument(
        "--progress_memory_top_tasks",
        type=int,
        default=None,
        help="How many retrieved similar tasks to use when generating milestones",
    )
    args = parser.parse_args()

    assert args.num_trials > 0, "Number of trials should be positive"
    assert args.num_envs > 0, "Number of environments should be positive"

    return args

def _init_env_configs(num_envs: int) -> List[Dict[str, Any]]:
    env_configs: List[Dict[str, Any]] = []
    for i in range(int(num_envs)):
        env_configs.append(
            {
                "name": f"env_{i}",
                "memory": [],
                "is_success": False,
                "skip": False,
            }
        )
    return env_configs


def _parse_env_statuses(trial_log_text: str) -> Dict[int, str]:
    """Parse 'STATUS: OK/FAIL' blocks from a trial log."""
    if not trial_log_text:
        return {}
    pattern = re.compile(
        r"Environment\s+#(\d+)\s*:\s*[\s\S]*?\nSTATUS:\s*(OK|FAIL)\s*$",
        flags=re.MULTILINE,
    )
    statuses: Dict[int, str] = {}
    for env_id_str, status in pattern.findall(trial_log_text):
        try:
            env_id = int(env_id_str)
        except Exception:
            continue
        statuses[env_id] = status
    return statuses


def _load_env_configs_for_resume(
    *,
    resume_dir: str,
    trial_idx: int,
    num_envs: int,
) -> Tuple[List[Dict[str, Any]], str]:
    """Load env configs for resuming a (possibly partial) trial.

    Priority:
      1) env_results_trial_{trial_idx}.json (checkpoint file, if present)
      2) env_results_trial_{trial_idx-1}.json (previous completed trial, if present)
      3) fresh init
    Then apply statuses parsed from trial_{trial_idx}.log if it exists.
    """
    trial_env_configs_log_path = os.path.join(resume_dir, f"env_results_trial_{trial_idx}.json")
    prev_env_configs_log_path = os.path.join(resume_dir, f"env_results_trial_{trial_idx - 1}.json")
    trial_log_path = os.path.join(resume_dir, f"trial_{trial_idx}.log")

    env_configs: List[Dict[str, Any]]
    source = "init"
    if os.path.exists(trial_env_configs_log_path) and os.path.getsize(trial_env_configs_log_path) > 0:
        with open(trial_env_configs_log_path, "r") as rf:
            env_configs = json.load(rf)
        source = os.path.basename(trial_env_configs_log_path)
    elif trial_idx > 0 and os.path.exists(prev_env_configs_log_path):
        with open(prev_env_configs_log_path, "r") as rf:
            env_configs = json.load(rf)
        source = os.path.basename(prev_env_configs_log_path)
    else:
        env_configs = _init_env_configs(num_envs)

    # Ensure length matches; if not, fall back to init but keep any overlapping successes.
    if not isinstance(env_configs, list):
        env_configs = _init_env_configs(num_envs)
        source = "init"
    if len(env_configs) != int(num_envs):
        fixed = _init_env_configs(num_envs)
        for i in range(min(len(env_configs), len(fixed))):
            if isinstance(env_configs[i], dict) and env_configs[i].get("is_success"):
                fixed[i]["is_success"] = True
            if isinstance(env_configs[i], dict) and env_configs[i].get("skip"):
                fixed[i]["skip"] = True
        env_configs = fixed
        source = f"{source} (shape-fixed)"

    # Apply statuses from existing trial log (so we can resume even if checkpoint JSON was never written).
    if os.path.exists(trial_log_path) and os.path.getsize(trial_log_path) > 0:
        with open(trial_log_path, "r", encoding="utf-8", errors="replace") as rf:
            statuses = _parse_env_statuses(rf.read())
        for env_id, status in statuses.items():
            if 0 <= env_id < len(env_configs):
                # A STATUS line means this env already completed before the
                # interruption. Skip both prior successes and prior failures
                # so resume only runs missing envs instead of giving failed
                # tasks a second attempt.
                env_configs[env_id]["skip"] = True
                env_configs[env_id]["is_success"] = status == "OK"

        # If no checkpoint file exists yet, write one now so subsequent resumes are faster/cleaner.
        if not os.path.exists(trial_env_configs_log_path):
            try:
                with open(trial_env_configs_log_path, "w") as wf:
                    json.dump(env_configs, wf, indent=4)
            except Exception:
                pass

    return env_configs, source


def main(args) -> None:
    if args.is_resume:
        if not os.path.exists(args.resume_dir):
            raise ValueError(f"Resume directory `{args.resume_dir}` does not exist")
        logging_dir = args.resume_dir

        env_configs, source = _load_env_configs_for_resume(
            resume_dir=logging_dir,
            trial_idx=int(args.start_trial_num),
            num_envs=int(args.num_envs),
        )
    else:
        # Create the run directory
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)
        logging_dir = args.run_name

        # initialize environment configs
        env_configs = _init_env_configs(args.num_envs)
    
    world_log_path: str = os.path.join(logging_dir, 'world.log')

    # print start status to user
    if args.is_resume:
        print(f"""
    -----
    Resuming run with the following parameters:
    Run name: {logging_dir}
    Number of trials: {args.num_trials}
    Number of environments: {args.num_envs}
    Resume trial number: {args.start_trial_num}
    Loaded env configs from: {source}

    Sending all logs to `{logging_dir}`
    -----
    """)
    else:
        print(f"""
    -----
    Starting run with the following parameters:
    Run name: {logging_dir}
    Number of trials: {args.num_trials}
    Number of environments: {args.num_envs}
    Sending all logs to `{logging_dir}`
    -----
    """)

    # run trials
    trial_idx = args.start_trial_num
    while trial_idx < args.num_trials:
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** Start Trial #{trial_idx} *****\n\n')

        # set paths to log files
        trial_log_path = os.path.join(logging_dir, f'trial_{trial_idx}.log')
        trial_env_configs_log_path = os.path.join(logging_dir, f'env_results_trial_{trial_idx}.json')
        if not args.is_resume:
            if os.path.exists(trial_log_path):
                open(trial_log_path, 'w').close()
            if os.path.exists(trial_env_configs_log_path):
                open(trial_env_configs_log_path, 'w').close()

        # run trial
        run_trial(
            trial_log_path,
            world_log_path,
            trial_idx,
            env_configs,
            args.model,
            progress_memory=args.progress_memory,
            progress_memory_library_path=args.progress_memory_library,
            progress_memory_top_tasks=args.progress_memory_top_tasks,
            use_local_fewshot=args.progress_memory,
            feasibility_memory=args.feasibility_memory,
            task_file=args.task_file,
            checkpoint_env_configs_path=trial_env_configs_log_path,
            resume=args.is_resume,
        )

        # log env configs for trial
        with open(trial_env_configs_log_path, 'w') as wf:
            json.dump(env_configs, wf, indent=4)

        # log world for trial
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** End Trial #{trial_idx} *****\n\n')

        trial_idx += 1


if __name__ == '__main__':
    args = get_args()
    main(args)
