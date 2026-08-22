#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-all}"
OUT="${DUALMEMORY_OUTPUT_DIR:-$ROOT/runs/s1_build}"

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing Stage 1 output: $path" >&2
    echo "Run Stage 1 first with: bash scripts/run_s1.sh <environment>" >&2
    exit 1
  fi
}

run_alfworld() {
  local base="$OUT/alfworld"
  local io="$base/io"
  require_dir "$io"
  require_dir "$base/s1_online_50"
  cd "$ROOT/dualmemory/alfworld/alfworld_runs"
  export DUALMEMORY_ALFWORLD_DIR="$io"
  python s2_main.py \
    --run_dir "$base/s1_online_50" \
    --output_path "$base/progress_memory_library.json"
}

run_webshop() {
  local base="$OUT/webshop"
  local io="$base/io"
  require_dir "$io"
  require_dir "$base/s1_online_50"
  cd "$ROOT/dualmemory/webshop/our_design"
  export DUALMEMORY_WEBSHOP_DIR="$io"
  python s2_main.py \
    --run_dir "$base/s1_online_50" \
    --output_path "$base/progress_memory_library.json"
}

run_textcraft() {
  local base="$OUT/textcraft"
  local io="$base/io"
  require_dir "$io"
  require_dir "$base/s1_traj"
  cd "$ROOT/dualmemory/textcraft/our_design"
  PYTHONHASHSEED=0 python s2_main.py \
    --io_dir "$io" \
    --traj_dir "$base/s1_traj" \
    --output_path "$base/progress_memory_library.json"
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
