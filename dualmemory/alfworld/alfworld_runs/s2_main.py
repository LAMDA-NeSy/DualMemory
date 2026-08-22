import argparse
import json
import os
import re
import time
from typing import Any, Dict, List

from progress_memory import (
    _embed_texts,
    _is_sentence_transformer_model,
    _normalize,
    extract_milestones,
    parse_env_history,
    parse_trial_log,
)
from api_config import apply_api_config, get_api_model
from utils import Model


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: Build Progress Memory library")
    parser.add_argument(
        "--source",
        choices=("run_dir", "buffer_traj"),
        default="run_dir",
        help="Build the Stage 2 library from Stage 1 trial logs or buffered trajectories",
    )
    parser.add_argument("--run_dir", default="", help="Directory with s1 trial_*.log files (required when --source=run_dir)")
    parser.add_argument(
        "--buffer_traj_dir",
        default="",
        help="Directory with buffer trajectories (defaults to io_dir/traj_data/alfworld/buffer_traj)",
    )
    parser.add_argument("--output_path", default="", help="Output progress memory library path (JSON)")
    parser.add_argument(
        "--model",
        type=str,
        default=get_api_model("milestone_extraction", "gpt-4o-2024-08-06"),
        help="LLM model for milestone extraction",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-mpnet-base-v2",
        help="SentenceTransformers embedding model for task/milestone retrieval (PROGRESS_MEMORY uses all-mpnet-base-v2)",
    )
    parser.add_argument("--max_tasks", type=int, default=0, help="Max number of successful tasks to process (0 = all)")
    parser.add_argument("--prompt_dir", type=str, default="", help="Prompt directory (defaults to io_dir/prompts)")
    parser.add_argument(
        "--log_every_tasks",
        type=int,
        default=10,
        help="Print progress every N successful tasks (0 = disable)",
    )
    return parser.parse_args()


def _default_io_dir() -> str:
    env_dir = os.environ.get("DUALMEMORY_ALFWORLD_DIR")
    if env_dir:
        return env_dir
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_prompt_dir() -> str:
    io_dir = _default_io_dir()
    return os.path.join(io_dir, "prompts")


def _default_output_path() -> str:
    io_dir = _default_io_dir()
    env_name = "alfworld"
    return os.path.join(io_dir, "traj_data", env_name, "progress_memory_library.json")


def _default_buffer_traj_dir() -> str:
    io_dir = _default_io_dir()
    env_name = "alfworld"
    return os.path.join(io_dir, "traj_data", env_name, "buffer_traj")


_INVALID_OBSERVATIONS = {"Nothing happens.", "OK."}


def _filter_invalid_observation_steps(parsed: Dict[str, Any]) -> Dict[str, Any]:
    trajectory = list(parsed.get("trajectory") or [])
    filtered = [
        step
        for step in trajectory
        if str(step.get("observation", "")).strip() not in _INVALID_OBSERVATIONS
    ]
    parsed = dict(parsed)
    parsed["trajectory"] = filtered
    parsed["actions"] = [step.get("action", "") for step in filtered]
    return parsed


def _segment_with_context(
    trajectory: List[Dict[str, str]],
    *,
    start_idx: int,
    end_idx: int,
    context_steps: int = 1,
) -> List[Dict[str, str]]:
    """Slice a milestone segment with optional context before/after."""
    if not trajectory:
        return []
    n = len(trajectory)
    a = max(0, min(int(start_idx), n - 1))
    b = max(0, min(int(end_idx), n - 1))
    if b < a:
        a, b = b, a
    c = max(0, int(context_steps))
    ctx_start = max(0, a - c)
    ctx_end = min(n - 1, b + c)
    return [trajectory[i] for i in range(ctx_start, ctx_end + 1)]


def _iter_buffer_episode_texts(text: str) -> List[str]:
    """Extract per-task episodes from buffer_traj text dumps.

    buffer_traj files are plain text logs that may contain multiple tasks back-to-back.
    Episodes can begin with either:
    - a line exactly like "Here is the task." / "Here is the task:"
    - a line like "Your task is to: ..."
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    task_header_re = re.compile(r"^\s*Here is the task[.:]?\s*$", re.IGNORECASE)
    your_task_re = re.compile(r"^\s*Your task is to\s*:\s*.*$", re.IGNORECASE)

    episodes: List[str] = []
    cur: List[str] = []

    for line in lines:
        if task_header_re.match(line) or your_task_re.match(line):
            if cur:
                episodes.append("\n".join(cur).strip())
                cur = []
            cur.append(line)
            continue
        if cur:
            cur.append(line)

    if cur:
        episodes.append("\n".join(cur).strip())

    normalized: List[str] = []
    for ep in episodes:
        if not ep:
            continue
        if "Here is the task" not in ep:
            ep = "Here is the task.\n" + ep
        normalized.append(ep)
    return normalized


def _finalize_and_save_library(
    tasks: List[Dict[str, Any]],
    output_path: str,
    embedding_model: str,
    start_time: float,
) -> Dict[str, Any]:
    if not tasks:
        raise RuntimeError("No successful trajectories found for milestone extraction.")

    total_milestones = sum(len(t.get("milestones", [])) for t in tasks)
    print(f"[s2] extracted {len(tasks)} tasks, {total_milestones} milestones", flush=True)

    # 计算task的embedding，归一化方便计算余弦相似度
    print("[s2] embedding tasks ...", flush=True)
    task_embeddings = _embed_texts([t["task"] for t in tasks], model=embedding_model)
    task_embeddings = [_normalize(vec) for vec in task_embeddings]

    # 计算milestone的embedding，归一化方便计算余弦相似度
    milestone_texts = [
        milestone["milestone"] for task in tasks for milestone in task["milestones"]
    ]
    print(f"[s2] embedding milestones ({len(milestone_texts)} texts) ...", flush=True)
    milestone_embeddings = _embed_texts(milestone_texts, model=embedding_model)
    milestone_embeddings = [_normalize(vec) for vec in milestone_embeddings]

    # 按顺序把他们塞回tasks字典
    idx = 0
    for task, embedding in zip(tasks, task_embeddings):
        task["task_embedding"] = embedding
        for milestone in task["milestones"]:
            milestone["milestone_embedding"] = milestone_embeddings[idx]
            idx += 1

    library = {
        "version": 1,
        "embedding_model": embedding_model,
        "tasks": tasks,
    }

    out_dir = os.path.dirname(output_path) or "."
    print(f"[s2] saving progress memory library -> {output_path}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2)
    elapsed = time.time() - start_time
    print(f"[s2] done in {elapsed:.1f}s", flush=True)
    return library


def build_library(
    run_dir: str,
    output_path: str,
    prompt_dir: str,
    model: Model,
    embedding_model: str,
    max_tasks: int = 0,
    log_every_tasks: int = 10,
) -> Dict[str, Any]:
    start_time = time.time()
    print(f"[s2] run_dir={run_dir}", flush=True)
    print(f"[s2] output_path={output_path}", flush=True)
    print(f"[s2] prompt_dir={prompt_dir}", flush=True)
    print(f"[s2] milestone_extraction_model={model}", flush=True)
    print(f"[s2] embedding_model={embedding_model}", flush=True)

    tasks: List[Dict[str, Any]] = []

    trial_files = sorted(
        f for f in os.listdir(run_dir) if f.startswith("trial_") and f.endswith(".log")
    )
    if not trial_files:
        raise FileNotFoundError(f"No trial_*.log files found in {run_dir}")
    print(f"[s2] found {len(trial_files)} trial log files", flush=True)

    total_env_records = 0
    total_ok_records = 0
    total_parsed_ok = 0

    # 遍历trial日志文件
    for trial_idx, trial_file in enumerate(trial_files, start=1):
        trial_path = os.path.join(run_dir, trial_file)
        print(f"[s2] ({trial_idx}/{len(trial_files)}) reading {trial_path}", flush=True)
        with open(trial_path, "r", encoding="utf-8") as f:
            text = f.read()

        trial_id = os.path.splitext(trial_file)[0]
        
        # parse trial log会把一个trail.log 变成
        # {"env_id": env_id,"env_history": env_history,"status": status}的list
        # 每一个表示一个task的记录
        env_records = parse_trial_log(text)
        total_env_records += len(env_records)
        for env_record in env_records:
            # 只用成功的task建立progress memory library
            if env_record["status"] != "OK":
                continue
            total_ok_records += 1
            parsed = parse_env_history(env_record["env_history"])
            """
            Here is the task: Put a cool apple in the fridge.
            > go to countertop 1
            On the countertop 1, you see a apple 1.
            > take apple 1
            You pick up the apple 1 from the countertop 1.
            > go to fridge 1
            The fridge 1 is closed.

            变成了 
{
    # 提取出的任务目标
    "task": "Put a cool apple in the fridge.",
    
    # 完整的轨迹列表，包含每一步的动作和对应的环境反馈
    "trajectory": [
        {
            "action": "go to countertop 1",
            "observation": "On the countertop 1, you see a apple 1."
        },
        {
            "action": "take apple 1",
            "observation": "You pick up the apple 1 from the countertop 1."
        },
        # ... 更多步骤
    ],
    
    # 纯动作列表（只包含动作字符串，方便后续处理）
    "actions": [
        "go to countertop 1",
        "take apple 1",
        # ...
    ]
}
...
            """
            if not parsed or not parsed.get("actions"):
                continue
            parsed = _filter_invalid_observation_steps(parsed)
            if not parsed.get("actions"):
                continue
            total_parsed_ok += 1

            # 用整段动作序列来分段/标注 milestone (与PROGRESS_MEMORY论文对齐)
            milestones = extract_milestones(
                task=parsed["task"],
                actions=parsed["actions"],
                model=model,
                prompt_dir=prompt_dir,
            )
            # milestones格式
            # [
            #     {
            #         "milestone": "Find the apple",
            #         "actions": [0, 1]
            #     },
            #     {
            #         "milestone": "Go to the fridge",
            #         "actions": [2]
            #     }
            # ]
            if not milestones:
                continue

            task_id = f"{trial_id}_env_{env_record['env_id']}"
            trajectory = parsed["trajectory"]
            milestone_entries = []

            # 遍历里程碑
            for idx, milestone in enumerate(milestones):
                raw_action_indices = sorted(
                    {i for i in milestone.get("actions", []) if 0 <= i < len(trajectory)}
                )
                if not raw_action_indices:
                    continue
                # 提取该里程碑的动作的start_idx和end_idx, 获得对应的连续动作片段
                start_idx = raw_action_indices[0]
                end_idx = raw_action_indices[-1]
                action_indices = list(range(start_idx, end_idx + 1))
                # Include previous + next step so the segment carries local context.
                segment = _segment_with_context(
                    trajectory, start_idx=start_idx, end_idx=end_idx, context_steps=1
                )
                # 组装单个里程碑
                milestone_entries.append(
                    {
                        "milestone_id": f"{task_id}_m{idx}",
                        "milestone": milestone["milestone"],
                        "order": idx,
                        "action_indices": action_indices,
                        "segment": segment,
                    }
                )

            if not milestone_entries:
                continue
            # 组装单个task的里程碑
            tasks.append(
                {
                    "task_id": task_id,
                    "task": parsed["task"],
                    "trajectory": trajectory,
                    "milestone_guide": [m["milestone"] for m in milestone_entries],
                    "milestones": milestone_entries,
                }
            )

            if log_every_tasks and (len(tasks) % log_every_tasks == 0):
                print(
                    f"[s2] processed {len(tasks)} successful tasks "
                    f"(env_records={total_env_records}, ok={total_ok_records}, parsed_ok={total_parsed_ok})",
                    flush=True,
                )

            if max_tasks and len(tasks) >= max_tasks:
                break
        if max_tasks and len(tasks) >= max_tasks:
            break

    return _finalize_and_save_library(
        tasks,
        output_path=output_path,
        embedding_model=embedding_model,
        start_time=start_time,
    )


def build_library_from_buffer_traj(
    buffer_traj_dir: str,
    output_path: str,
    prompt_dir: str,
    model: Model,
    embedding_model: str,
    max_tasks: int = 0,
    log_every_tasks: int = 10,
) -> Dict[str, Any]:
    start_time = time.time()
    print(f"[s2] buffer_traj_dir={buffer_traj_dir}", flush=True)
    print(f"[s2] output_path={output_path}", flush=True)
    print(f"[s2] prompt_dir={prompt_dir}", flush=True)
    print(f"[s2] milestone_extraction_model={model}", flush=True)
    print(f"[s2] embedding_model={embedding_model}", flush=True)

    if not os.path.isdir(buffer_traj_dir):
        raise FileNotFoundError(f"buffer_traj_dir not found: {buffer_traj_dir}")

    tasks: List[Dict[str, Any]] = []

    traj_dirs = sorted(
        d
        for d in os.listdir(buffer_traj_dir)
        if d.startswith("traj_") and os.path.isdir(os.path.join(buffer_traj_dir, d))
    )
    if not traj_dirs:
        raise FileNotFoundError(f"No traj_* directories found in {buffer_traj_dir}")
    print(f"[s2] found {len(traj_dirs)} traj dirs", flush=True)

    total_episode_records = 0
    total_parsed_ok = 0

    for traj_idx, traj_dir in enumerate(traj_dirs, start=1):
        dir_path = os.path.join(buffer_traj_dir, traj_dir)
        files = sorted(f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)))
        if not files:
            continue

        for fname in files:
            fpath = os.path.join(dir_path, fname)
            print(f"[s2] ({traj_idx}/{len(traj_dirs)}) reading {fpath}", flush=True)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()

            episodes = _iter_buffer_episode_texts(text)
            total_episode_records += len(episodes)

            file_stem = os.path.splitext(fname)[0]
            for ep_idx, env_history in enumerate(episodes):
                parsed = parse_env_history(env_history)
                if not parsed or not parsed.get("actions"):
                    continue
                parsed = _filter_invalid_observation_steps(parsed)
                if not parsed.get("actions"):
                    continue
                total_parsed_ok += 1

                milestones = extract_milestones(
                    task=parsed["task"],
                    actions=parsed["actions"],
                    model=model,
                    prompt_dir=prompt_dir,
                )
                if not milestones:
                    continue

                task_id = f"buffer_{traj_dir}_{file_stem}_ep{ep_idx}"
                trajectory = parsed["trajectory"]
                milestone_entries = []

                for idx, milestone in enumerate(milestones):
                    raw_action_indices = sorted(
                        {i for i in milestone.get("actions", []) if 0 <= i < len(trajectory)}
                    )
                    if not raw_action_indices:
                        continue
                    start_idx = raw_action_indices[0]
                    end_idx = raw_action_indices[-1]
                    action_indices = list(range(start_idx, end_idx + 1))
                    segment = _segment_with_context(
                        trajectory, start_idx=start_idx, end_idx=end_idx, context_steps=1
                    )
                    milestone_entries.append(
                        {
                            "milestone_id": f"{task_id}_m{idx}",
                            "milestone": milestone["milestone"],
                            "order": idx,
                            "action_indices": action_indices,
                            "segment": segment,
                        }
                    )

                if not milestone_entries:
                    continue

                tasks.append(
                    {
                        "task_id": task_id,
                        "task": parsed["task"],
                        "trajectory": trajectory,
                        "milestone_guide": [m["milestone"] for m in milestone_entries],
                        "milestones": milestone_entries,
                    }
                )

                if log_every_tasks and (len(tasks) % log_every_tasks == 0):
                    print(
                        f"[s2] processed {len(tasks)} tasks "
                        f"(episodes={total_episode_records}, parsed_ok={total_parsed_ok})",
                        flush=True,
                    )

                if max_tasks and len(tasks) >= max_tasks:
                    break

            if max_tasks and len(tasks) >= max_tasks:
                break

        if max_tasks and len(tasks) >= max_tasks:
            break

    print(
        f"[s2] parsed episodes: total={total_episode_records}, parsed_ok={total_parsed_ok}",
        flush=True,
    )

    return _finalize_and_save_library(
        tasks,
        output_path=output_path,
        embedding_model=embedding_model,
        start_time=start_time,
    )




def main() -> None:
    args = get_args()
    apply_api_config()

    if not _is_sentence_transformer_model(args.embedding_model):
        raise ValueError(
            "S2 progress memory library embedding must use SentenceTransformers (no embedding API). "
            f"Got embedding_model={args.embedding_model!r}. "
            "Try --embedding_model all-mpnet-base-v2 (PROGRESS_MEMORY default)."
        )

    output_path = args.output_path or _default_output_path()
    prompt_dir = args.prompt_dir or _default_prompt_dir()

    if args.source == "run_dir":
        if not args.run_dir:
            raise SystemExit("--run_dir is required when --source=run_dir")
        build_library(
            run_dir=args.run_dir,
            output_path=output_path,
            prompt_dir=prompt_dir,
            model=args.model,
            embedding_model=args.embedding_model,
            max_tasks=args.max_tasks,
            log_every_tasks=args.log_every_tasks,
        )
    elif args.source == "buffer_traj":
        buffer_traj_dir = args.buffer_traj_dir or _default_buffer_traj_dir()
        build_library_from_buffer_traj(
            buffer_traj_dir=buffer_traj_dir,
            output_path=output_path,
            prompt_dir=prompt_dir,
            model=args.model,
            embedding_model=args.embedding_model,
            max_tasks=args.max_tasks,
            log_every_tasks=args.log_every_tasks,
        )
    else:
        raise SystemExit(f"Unknown --source: {args.source}")
    print(f"Progress memory library saved to {output_path}")


if __name__ == "__main__":
    main()
