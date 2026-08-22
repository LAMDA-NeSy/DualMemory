import argparse
import json
import os
import time
from typing import Any, Dict, List

from api_config import apply_api_config, get_api_model
from progress_memory import (
    _embed_texts,
    _is_sentence_transformer_model,
    _normalize,
    extract_milestones,
    parse_env_history,
    parse_trial_log,
)


def _default_io_dir() -> str:
    env_dir = os.environ.get("DUALMEMORY_WEBSHOP_DIR")
    if env_dir:
        return env_dir
    return os.path.abspath(os.path.dirname(__file__))


def _default_prompt_dir(io_dir: str) -> str:
    return os.path.join(io_dir, "prompts")


def _default_output_path(io_dir: str) -> str:
    return os.path.join(io_dir, "symbolic_knowledge", "webshop", "library.json")


def _resolve_run_dir(run_dir: str) -> str:
    if not run_dir:
        raise ValueError("run_dir must be non-empty")
    if os.path.isabs(run_dir):
        return run_dir
    base_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_dir, run_dir)


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


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: Build WebShop Progress Memory library.")
    parser.add_argument("--run_dir", default="", help="Directory with s1 trial_*.log files (relative to webshop/our_design if not absolute)")
    parser.add_argument("--output_path", default="", help="Output progress memory library path (JSON)")
    parser.add_argument(
        "--model",
        type=str,
        default=get_api_model("milestone_extraction", "gpt-4o-mini"),
        help="LLM model for milestone extraction",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-mpnet-base-v2",
        help="SentenceTransformers embedding model (default: all-mpnet-base-v2)",
    )
    parser.add_argument("--max_tasks", type=int, default=0, help="Max number of successful tasks to process (0=all)")
    parser.add_argument("--prompt_dir", type=str, default="", help="Prompt directory (defaults to io_dir/prompts)")
    parser.add_argument("--log_every_tasks", type=int, default=5)
    return parser.parse_args()


def _finalize_and_save_library(
    tasks: List[Dict[str, Any]],
    output_path: str,
    embedding_model: str,
    start_time: float,
) -> Dict[str, Any]:
    if not tasks:
        raise RuntimeError("No successful trajectories found for milestone extraction.")

    total_milestones = sum(len(t.get("milestones", [])) for t in tasks)
    print(f"[s2-webshop] extracted {len(tasks)} tasks, {total_milestones} milestones", flush=True)

    print("[s2-webshop] embedding tasks ...", flush=True)
    task_embeddings = _embed_texts([t["task"] for t in tasks], model=embedding_model)
    task_embeddings = [_normalize(vec) for vec in task_embeddings]

    milestone_texts = [m["milestone"] for t in tasks for m in t["milestones"]]
    print(f"[s2-webshop] embedding milestones ({len(milestone_texts)} texts) ...", flush=True)
    milestone_embeddings = _embed_texts(milestone_texts, model=embedding_model)
    milestone_embeddings = [_normalize(vec) for vec in milestone_embeddings]

    idx = 0
    for task, emb in zip(tasks, task_embeddings):
        task["task_embedding"] = emb
        for milestone in task["milestones"]:
            milestone["milestone_embedding"] = milestone_embeddings[idx]
            idx += 1

    library = {"version": 1, "embedding_model": embedding_model, "tasks": tasks}

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2)
    print(f"[s2-webshop] saved -> {output_path}", flush=True)
    print(f"[s2-webshop] done in {time.time() - start_time:.1f}s", flush=True)
    return library


def build_library(
    run_dir: str,
    output_path: str,
    prompt_dir: str,
    model: str,
    embedding_model: str,
    max_tasks: int = 0,
    log_every_tasks: int = 10,
) -> Dict[str, Any]:
    start_time = time.time()
    tasks: List[Dict[str, Any]] = []

    trial_files = sorted(f for f in os.listdir(run_dir) if f.startswith("trial_") and f.endswith(".log"))
    if not trial_files:
        raise FileNotFoundError(f"No trial_*.log files found in {run_dir}")

    # 实际也就一个trial_0.log
    for trial_idx, trial_file in enumerate(trial_files, start=1):
        trial_path = os.path.join(run_dir, trial_file)
        print(f"[s2-webshop] ({trial_idx}/{len(trial_files)}) reading {trial_path}", flush=True)
        with open(trial_path, "r", encoding="utf-8") as f:
            text = f.read()
        trial_id = os.path.splitext(trial_file)[0]

        # 解析这个trial的task记录，返回{"env_id", "status", "env_history"}
        env_records = parse_trial_log(text)
        for env_record in env_records:
            # 只看成功的轨迹
            if env_record.get("status") != "OK":
                continue
            parsed = parse_env_history(env_record.get("env_history", ""))
            if not parsed or not parsed.get("actions"):
                continue
            # 提取milestones
            milestones = extract_milestones(
                task=parsed["task"],
                actions=parsed["actions"],
                model=model,
                prompt_dir=prompt_dir,
            )
            if not milestones:
                continue

            task_id = f"{trial_id}_env_{env_record['env_id']}"
            trajectory = parsed["trajectory"]
            milestone_entries = []
            for idx, m in enumerate(milestones):
                raw = sorted({i for i in m.get("actions", []) if 0 <= i < len(trajectory)})
                if not raw:
                    continue
                start_i, end_i = raw[0], raw[-1]
                action_indices = list(range(start_i, end_i + 1))
                # Include previous + next step so the segment carries local context.
                segment = _segment_with_context(trajectory, start_idx=start_i, end_idx=end_i, context_steps=1)
                milestone_entries.append(
                    {
                        "milestone_id": f"{task_id}_m{idx}",
                        "milestone": m["milestone"],
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
                print(f"[s2-webshop] processed {len(tasks)} successful tasks", flush=True)

            if max_tasks and len(tasks) >= max_tasks:
                break
        if max_tasks and len(tasks) >= max_tasks:
            break

    return _finalize_and_save_library(tasks, output_path=output_path, embedding_model=embedding_model, start_time=start_time)


def main() -> None:
    args = get_args()
    apply_api_config()

    io_dir = _default_io_dir()
    if not args.run_dir:
        raise SystemExit("--run_dir is required (s1 run directory containing trial_*.log)")
    run_dir = _resolve_run_dir(args.run_dir)

    prompt_dir = args.prompt_dir or _default_prompt_dir(io_dir)
    output_path = args.output_path or _default_output_path(io_dir)

    if not _is_sentence_transformer_model(args.embedding_model):
        raise ValueError(
            "Progress memory library embedding must use SentenceTransformers (offline). "
            f"Got embedding_model={args.embedding_model!r}."
        )

    build_library(
        run_dir=run_dir,
        output_path=output_path,
        prompt_dir=prompt_dir,
        model=args.model,
        embedding_model=args.embedding_model,
        max_tasks=args.max_tasks,
        log_every_tasks=args.log_every_tasks,
    )


if __name__ == "__main__":
    main()
