import argparse
import json
import os
import re
from pathlib import Path


def _default_io_dir() -> str:
    env_dir = os.environ.get("DUALMEMORY_ALFWORLD_DIR")
    if env_dir:
        return env_dir
    # This file lives at: alfworld_runs/stateinfo_transform/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


_LOC_LINE_RE = re.compile(r"looking quickly around you, you see", flags=re.IGNORECASE)
_SEE_RE = re.compile(r"you see", flags=re.IGNORECASE)
_TYPE_NUM_IN_SEE_RE = re.compile(r"\b(?:a|an)\s+([a-zA-Z]+)\s+\d+\b")
_TYPE_NUM_IN_TAKE_RE = re.compile(r"^>\s*take\s+([a-zA-Z]+)\s+\d+\b", flags=re.IGNORECASE)
_TASK_LINE_RE = re.compile(r"^your task is to:\s*(.+)$", flags=re.IGNORECASE)
_WORD_RE = re.compile(r"[a-zA-Z]+")

_TASK_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "all",
    "be",
    "both",
    "cold",
    "cool",
    "clean",
    "dirty",
    "examine",
    "find",
    "first",
    "heat",
    "hot",
    "in",
    "into",
    "it",
    "look",
    "on",
    "or",
    "pick",
    "place",
    "put",
    "second",
    "some",
    "take",
    "them",
    "then",
    "the",
    "to",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "up",
    "under",
    "inside",
    "onto",
    "off",
    "with",
}


def extract_vocab_from_text(text: str) -> tuple[set[str], set[str]]:
    """
    Returns:
      - location_types: types mentioned in the global room "Looking quickly..." line(s)
      - object_types: types seen in 'you see ...' lists or in 'take X N from ...' actions
    """
    location_types: set[str] = set()
    object_types: set[str] = set()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        lower = line.lower()

        if _LOC_LINE_RE.search(lower):
            after = lower.split("you see", 1)[-1]
            location_types.update(_TYPE_NUM_IN_SEE_RE.findall(after))

        if _SEE_RE.search(lower):
            after = lower.split("you see", 1)[-1]
            object_types.update(_TYPE_NUM_IN_SEE_RE.findall(after))

        m = _TYPE_NUM_IN_TAKE_RE.match(line)
        if m:
            object_types.add(m.group(1).lower())

        tm = _TASK_LINE_RE.match(line)
        if tm:
            for w in _WORD_RE.findall(tm.group(1).lower()):
                if w in _TASK_STOPWORDS:
                    continue
                object_types.add(w)

    return {t.lower() for t in location_types}, {t.lower() for t in object_types}


def build_item_list(traj_root: Path) -> list[str]:
    if not traj_root.exists():
        raise FileNotFoundError(f"traj_root does not exist: {traj_root}")

    location_types: set[str] = set()
    object_types: set[str] = set()

    files = sorted(traj_root.glob("traj_*/**/*.json"))
    if not files:
        raise RuntimeError(f"No trajectory files found under: {traj_root}")

    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        loc, obj = extract_vocab_from_text(text)
        location_types.update(loc)
        object_types.update(obj)

    item_types = sorted(object_types - location_types)
    return item_types


def main() -> None:
    parser = argparse.ArgumentParser(description="Build item whitelist from buffer_traj logs.")
    parser.add_argument("--io_dir", default="", help="ALFWorld io_dir (defaults to DUALMEMORY_ALFWORLD_DIR or repo root).")
    parser.add_argument("--env_name", default="alfworld", help="Environment name under traj_data/ (default: alfworld).")
    parser.add_argument(
        "--traj_root",
        default="",
        help="Explicit buffer_traj path (overrides io_dir/env_name default).",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output JSON path (default: stateinfo_transform/item_list.json).",
    )
    args = parser.parse_args()

    io_dir = args.io_dir or _default_io_dir()
    traj_root = Path(args.traj_root) if args.traj_root else Path(io_dir) / "traj_data" / args.env_name / "buffer_traj"

    out_path = Path(args.out) if args.out else Path(__file__).with_name("item_list.json")
    items = build_item_list(traj_root)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(items)} items to {out_path}")


if __name__ == "__main__":
    main()
