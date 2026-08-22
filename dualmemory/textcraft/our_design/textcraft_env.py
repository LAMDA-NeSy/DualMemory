from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _is_repo_textcraft_module(mod: ModuleType) -> bool:
    mod_file = getattr(mod, "__file__", "") or ""
    if not mod_file:
        return False
    try:
        p = Path(mod_file).resolve()
    except Exception:
        return False
    # If we are importing this repository's TextCraft method package, it is
    # shadowing the pip package that provides TextCraft.
    return "dualmemory" in p.parts and "textcraft" in p.parts


def get_textcraft_env():
    """
    Import `TextCraft` from the *pip package* `textcraft`.

    This repo also has a local TextCraft method package which can shadow the
    pip package when running scripts from the repo root. We defensively retry the import
    after temporarily removing the repo root from sys.path.
    """
    try:
        from textcraft import TextCraft  # type: ignore

        return TextCraft
    except Exception:
        pass

    # Retry: temporarily remove likely repo roots from sys.path
    old_sys_path = list(sys.path)
    try:
        # Remove "" (cwd) and any parent that looks like this repo root.
        cleaned: list[str] = []
        for entry in sys.path:
            if entry in ("", "."):
                continue
            try:
                resolved = Path(entry).resolve()
            except Exception:
                resolved = Path(entry)
            if resolved.name == "DualMemory":
                continue
            cleaned.append(entry)
        sys.path = cleaned

        import textcraft as tc  # type: ignore

        TextCraft = getattr(tc, "TextCraft", None)
        if TextCraft is not None:
            return TextCraft

        raise ImportError(
            "Imported a `textcraft` module but it does not expose `TextCraft`.\n"
            "This usually means the repo-local `textcraft/` directory is shadowing the pip package.\n"
            "Fix:\n"
            "1) `pip install -r requirements.txt`\n"
            "2) Run from `dualmemory/textcraft/our_design/` (recommended).\n"
            "3) Then run a script, e.g. `python s3_main.py ...`"
        )
    except ModuleNotFoundError as e:
        raise ImportError(
            "Missing pip package `textcraft`.\n"
            "Fix:\n"
            "1) `pip install -r requirements.txt`\n"
            "2) Run scripts from `dualmemory/textcraft/our_design/`.\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e
    finally:
        sys.path = old_sys_path
