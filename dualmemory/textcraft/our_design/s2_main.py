from __future__ import annotations

import argparse
from pathlib import Path

from api_config import apply_api_config
from api_config import get_api_model
from progress_memory import _is_sentence_transformer_model
from milestones import build_progress_memory_library
from utils import default_io_dir


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: Build TextCraft progress memory library (PROGRESS_MEMORY-style) from successful trajectories.")
    p.add_argument(
        "--io_dir",
        default="",
        help="textcraft/our_design io_dir (default: DUALMEMORY_TEXTCRAFT_DIR or this folder).",
    )
    p.add_argument(
        "--traj_dir",
        default="",
        help="Trajectory root directory (buffer_traj). If relative, resolved under io_dir. "
        "Default: io_dir/traj_data/<env_name>/buffer_traj.",
    )
    p.add_argument("--env_name", default="textcraft")
    p.add_argument("--output_path", default="", help="Output progress memory library path (JSON).")
    p.add_argument("--output", default="", help="Alias of --output_path (kept for backward compatibility).")
    p.add_argument("--include_failures", action="store_true", help="Include failed episodes (not recommended).")
    p.add_argument("--model", dest="planner_model", default="", help="LLM model for milestone extraction (milestone + action indices).")
    p.add_argument("--planner_model", dest="planner_model", default="", help="Alias of --model (kept for backward compatibility).")
    p.add_argument(
        "--embedding_model",
        default=get_api_model("embedding", "all-mpnet-base-v2"),
        help="Embedding model for retrieval indexing. "
        "Current S2 implementation requires a SentenceTransformers model (default: all-mpnet-base-v2).",
    )
    return p.parse_args()


def _resolve_output_path(io_dir: str, output_path: str) -> str:
    out = (output_path or "").strip() or "progress_memory_library.json"
    p = Path(out)
    if not p.is_absolute():
        p = Path(io_dir) / p
    return str(p)


def main() -> None:
    args = get_args()
    apply_api_config()

    io_dir = (args.io_dir or "").strip() or default_io_dir()
    output_arg = (args.output_path or "").strip() or (args.output or "").strip()
    output_path = _resolve_output_path(io_dir, output_arg)

    emb_model = (args.embedding_model or "").strip()
    if emb_model and not _is_sentence_transformer_model(emb_model):
        raise SystemExit(
            "Progress memory library embedding must use SentenceTransformers (offline). "
            f"Got embedding_model={emb_model!r}."
        )

    library = build_progress_memory_library(
        io_dir=io_dir,
        env_name=args.env_name,
        traj_dir=(args.traj_dir or "").strip(),
        output_path=output_path,
        only_success=not bool(args.include_failures),
        planner_model=(args.planner_model or "").strip(),
        embedding_model=emb_model,
    )
    tasks = list(library.get("tasks") or [])
    total_milestones = sum(len(t.get("milestones") or []) for t in tasks if isinstance(t, dict))
    print(f"Saved progress memory library: {output_path} (tasks={len(tasks)}, milestones={total_milestones})")


if __name__ == "__main__":
    main()
