#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-all}"
OUT="${DUALMEMORY_OUTPUT_DIR:-$ROOT/runs/s3_eval}"
S1_OUT="${DUALMEMORY_S1_OUTPUT_DIR:-$ROOT/runs/s1_build}"
EVAL_RUNS="${DUALMEMORY_EVAL_RUNS:-3}"

if ! [[ "$EVAL_RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "DUALMEMORY_EVAL_RUNS must be a positive integer, got: $EVAL_RUNS" >&2
  exit 2
fi

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing generated artifact: $path" >&2
    echo "Run S1 and S2 first with: bash scripts/run_s1.sh <environment> && bash scripts/run_s2.sh <environment>" >&2
    exit 1
  fi
}

prepare_output_dir() {
  local path="$1"
  if [[ -e "$path" ]] && [[ -n "$(find "$path" -mindepth 1 -print -quit)" ]]; then
    echo "Stage 3 output directory is not empty: $path" >&2
    echo "Choose a new DUALMEMORY_OUTPUT_DIR or remove this previous Stage 3 output first." >&2
    exit 1
  fi
  mkdir -p "$path"
}

run_alfworld() {
  local base="$S1_OUT/alfworld"
  local io="$base/io"
  local library="$base/progress_memory_library.json"
  require_file "$library"
  require_file "$io/symbolic_knowledge/alfworld/pruned_rules_code.json"
  cd "$ROOT/dualmemory/alfworld/alfworld_runs"
  export DUALMEMORY_ALFWORLD_DIR="$io"
  for ((run_id = 1; run_id <= EVAL_RUNS; run_id++)); do
    local run_dir="$OUT/alfworld_r$run_id"
    prepare_output_dir "$run_dir"
    python s3_main.py \
      --num_trials 1 \
      --num_envs 134 \
      --run_name "$run_dir" \
      --task_file "$ROOT/tasks/alfworld_s3_tasks_suffix.json" \
      --progress_memory \
      --progress_memory_library "$library" \
      --feasibility_memory
  done
}

run_webshop() {
  local base="$S1_OUT/webshop"
  local io="$base/io"
  local library="$base/progress_memory_library.json"
  require_file "$library"
  require_file "$io/symbolic_knowledge/webshop/pruned_rules_code.json"
  python - <<'PY'
import socket
with socket.create_connection(("127.0.0.1", 3000), timeout=2):
    pass
PY
  cd "$ROOT/dualmemory/webshop/our_design"
  export DUALMEMORY_WEBSHOP_DIR="$io"
  for ((run_id = 1; run_id <= EVAL_RUNS; run_id++)); do
    local run_dir="$OUT/webshop_r$run_id"
    prepare_output_dir "$run_dir"
    python s3_main.py \
      --run_name "$run_dir" \
      --task_file "$ROOT/tasks/webshop_s3_fixed.json" \
      --progress_memory_library "$library" \
      --progress_memory \
      --feasibility_memory
  done
}

run_textcraft() {
  local base="$S1_OUT/textcraft"
  local io="$base/io"
  local library="$base/progress_memory_library.json"
  local rules="$io/symbolic_knowledge/textcraft/pruned_rules_code.json"
  require_file "$library"
  require_file "$rules"
  cd "$ROOT/dualmemory/textcraft/our_design"
  for ((run_id = 1; run_id <= EVAL_RUNS; run_id++)); do
    local run_dir="$OUT/textcraft_r$run_id"
    prepare_output_dir "$run_dir"
    PYTHONHASHSEED=0 python s3_main.py \
      --task_file "$ROOT/tasks/textcraft_s3_fixed.json" \
      --io_dir "$io" \
      --results_dir "$run_dir/results" \
      --traj_dir "$run_dir/traj" \
      --progress_memory \
      --progress_memory_library "$library" \
      --feasibility_memory \
      --rules_code_path "$rules"
  done
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
