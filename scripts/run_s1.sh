#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-all}"
OUT="${DUALMEMORY_OUTPUT_DIR:-$ROOT/runs/s1_build}"

require_empty_output_dir() {
  local path="$1"
  if [[ -e "$path" ]] && [[ -n "$(find "$path" -mindepth 1 -print -quit)" ]]; then
    echo "Stage 1 output directory is not empty: $path" >&2
    echo "Choose a new DUALMEMORY_OUTPUT_DIR or remove this previous Stage 1 output first." >&2
    exit 1
  fi
}

copy_prompts() {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst/prompts"
  cp -a "$src"/. "$dst/prompts"/
}

run_alfworld() {
  local base="$OUT/alfworld"
  local io="$base/io"
  require_empty_output_dir "$base"
  mkdir -p "$io/symbolic_knowledge/alfworld" "$io/traj_data/alfworld/buffer_traj" "$io/traj_data/alfworld/buffer_SG"
  copy_prompts "$ROOT/dualmemory/alfworld/prompts" "$io"
  cd "$ROOT/dualmemory/alfworld/alfworld_runs"
  export DUALMEMORY_ALFWORLD_DIR="$io"
  python s1_main.py \
    --num_trials 1 \
    --num_envs 50 \
    --run_name "$base/s1_online_50" \
    --task_file "$ROOT/tasks/alfworld_s1_tasks_suffix.json" \
    --online_rules
}

run_webshop() {
  local base="$OUT/webshop"
  local io="$base/io"
  require_empty_output_dir "$base"
  mkdir -p "$io/symbolic_knowledge/webshop" "$io/traj_data/webshop/buffer_traj" "$io/traj_data/webshop/buffer_SG"
  copy_prompts "$ROOT/dualmemory/webshop/our_design/prompts" "$io"
  cd "$ROOT/dualmemory/webshop/our_design"
  export DUALMEMORY_WEBSHOP_DIR="$io"
  python s1_main.py \
    --num_trials 1 \
    --run_name "$base/s1_online_50" \
    --task_file "$ROOT/tasks/webshop_s1_fixed.json" \
    --interval 5 \
    --online_rules
}

run_textcraft() {
  local base="$OUT/textcraft"
  local io="$base/io"
  require_empty_output_dir "$base"
  mkdir -p "$io/symbolic_knowledge/textcraft" "$io/traj_data/textcraft"
  copy_prompts "$ROOT/dualmemory/textcraft/our_design/prompts" "$io"
  cd "$ROOT/dualmemory/textcraft/our_design"
  PYTHONHASHSEED=0 python s1_main.py \
    --io_dir "$io" \
    --task_file "$ROOT/tasks/textcraft_s1_fixed.json" \
    --results_dir "$base/s1_results" \
    --traj_dir "$base/s1_traj"
}

case "$TARGET" in
  all)
    run_alfworld
    run_webshop
    run_textcraft
    ;;
  alfworld) run_alfworld ;;
  webshop) run_webshop ;;
  textcraft) run_textcraft ;;
  *)
    echo "Usage: $0 [all|alfworld|webshop|textcraft]" >&2
    exit 2
    ;;
esac
