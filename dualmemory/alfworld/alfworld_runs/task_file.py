import json
import os
from pathlib import Path
from typing import Any, Dict, List


def load_ordered_gamefiles(task_file: str, *, split: str) -> List[str]:
    """Load and validate the ordered ALFWorld gamefiles published with the repo."""
    path = Path(task_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ALFWorld task file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"ALFWorld task file must contain a non-empty JSON list: {path}")

    alfworld_data = os.environ.get("ALFWORLD_DATA", "").strip()
    if not alfworld_data:
        raise RuntimeError("ALFWORLD_DATA must be set when using an ALFWorld task file.")
    data_root = Path(alfworld_data).expanduser().resolve()

    expected_split = "train" if split == "train" else "valid_unseen"
    gamefiles: List[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Task entry {index} is not a JSON object.")
        gamefile = str(entry.get("gamefile", "")).strip()
        if not gamefile:
            raise ValueError(f"Task entry {index} has no gamefile.")

        marker = "data/alfworld/"
        if gamefile.startswith(marker):
            relative = gamefile[len(marker):]
        elif gamefile.startswith("data/"):
            relative = gamefile[len("data/"):]
        else:
            raise ValueError(
                f"Task entry {index} has unsupported gamefile path: {gamefile}"
            )

        relative_path = Path(relative)
        if expected_split not in relative_path.parts:
            raise ValueError(
                f"Task entry {index} is not in the expected ALFWorld split "
                f"{expected_split!r}: {gamefile}"
            )

        resolved = (data_root / relative_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"ALFWorld gamefile for task entry {index} does not exist: {resolved}"
            )
        gamefiles.append(str(resolved))

    return gamefiles


def init_ordered_alfworld_env(
    *,
    config: Dict[str, Any],
    environment_type: str,
    split: str,
    task_file: str,
    expected_num_envs: int,
):
    """Create AlfredTWEnv with the exact order from a published task file."""
    import alfworld.agents.environment

    env = alfworld.agents.environment.get_environment(environment_type)(
        config,
        train_eval=split,
    )
    selected_gamefiles = load_ordered_gamefiles(task_file, split=split)
    if len(selected_gamefiles) != int(expected_num_envs):
        raise ValueError(
            f"ALFWorld task file contains {len(selected_gamefiles)} tasks, "
            f"but the run requested {int(expected_num_envs)} environments."
        )
    env.game_files = selected_gamefiles
    env.num_games = len(selected_gamefiles)
    return env.init_env(batch_size=1)
