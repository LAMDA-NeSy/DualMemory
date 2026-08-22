import os
import warnings
from urllib.parse import urlparse
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - runtime dependency check
    yaml = None


_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _is_gpt_model_name(model_name: Optional[str]) -> bool:
    name = str(model_name or "").strip().lower()
    return "gpt" in name


def _default_config_path() -> str:
    env_path = os.environ.get("ALFWORLD_API_CONFIG_PATH")
    if env_path:
        return env_path
    return os.path.join(os.path.dirname(__file__), "api_config.yaml")


def load_api_config(path: Optional[str] = None, refresh: bool = False) -> Dict[str, Any]:
    global _CONFIG_CACHE
    if path is None and _CONFIG_CACHE is not None and not refresh:
        return _CONFIG_CACHE

    config_path = path or _default_config_path()
    if not os.path.exists(config_path):
        if path is None:
            _CONFIG_CACHE = {}
        return {}

    if yaml is None:
        raise RuntimeError("PyYAML is required to load ALFWorld API config.")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if path is None:
        _CONFIG_CACHE = data
    return data


def _get_api_section(config: Dict[str, Any]) -> Dict[str, Any]:
    api_section = config.get("api", {})
    if isinstance(api_section, dict):
        return api_section
    return {}


def get_api_setting(key: str, default: str = "") -> str:
    config = load_api_config()
    api_section = _get_api_section(config)
    value = api_section.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def get_api_model(key: str, default: str) -> str:
    config = load_api_config()
    models = config.get("models", {})
    value = models.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _looks_like_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def resolve_api_for_model(
    model_name: Optional[str] = None,
    *,
    path: Optional[str] = None,
    refresh: bool = False,
) -> Dict[str, str]:
    """
    Resolve API settings for a given model name.

    YAML supports either:
    - Legacy flat format:
        api: { api_key: ..., api_base: ... }
    - New split format (preferred):
        api:
          gpt: { api_key: ..., api_base: ... }   # used when model name contains "gpt"
          other: { api_key: ..., api_base: ... } # used otherwise
    """
    config = load_api_config(path=path, refresh=refresh)
    api_section = _get_api_section(config)

    api_key = ""
    api_base = ""

    # New format: api.gpt / api.other
    wants_profile = "gpt" if _is_gpt_model_name(model_name) else "other"
    profile_section = api_section.get(wants_profile) if isinstance(api_section, dict) else None
    if isinstance(profile_section, dict):
        api_key = str(profile_section.get("api_key") or "").strip()
        api_base = str(profile_section.get("api_base") or profile_section.get("base_url") or "").strip()

    # Legacy fallback: api.api_key / api.api_base
    if not api_key:
        api_key = str(api_section.get("api_key") or "").strip()
    if not api_base:
        api_base = str(api_section.get("api_base") or api_section.get("base_url") or "").strip()

    # Env fallback
    if not api_key:
        api_key = str(os.environ.get("OPENAI_API_KEY", "")).strip()
    if not api_base:
        api_base = str(os.environ.get("OPENAI_BASE_URL", "")).strip()

    # Common misconfiguration: swapping api_key and api_base in YAML.
    if api_key and api_base and _looks_like_url(api_key) and not _looks_like_url(api_base):
        warnings.warn(
            "ALFWorld api_config: resolved `api_key` looks like a URL and `api_base` does not; "
            "swapping them (api_key <-> api_base).",
            RuntimeWarning,
        )
        api_key, api_base = api_base, api_key

    return {"api_key": api_key, "api_base": api_base}


def apply_api_config(
    path: Optional[str] = None,
    refresh: bool = False,
    model_name: Optional[str] = None,
) -> Dict[str, str]:
    settings = resolve_api_for_model(model_name=model_name, path=path, refresh=refresh)
    api_key = settings.get("api_key", "")
    api_base = settings.get("api_base", "")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_BASE_URL"] = api_base
        os.environ["OPENAI_API_BASE"] = api_base

    try:
        import openai

        if api_key:
            openai.api_key = api_key
        if api_base:
            if hasattr(openai, "base_url"):
                openai.base_url = api_base
            if hasattr(openai, "api_base"):
                openai.api_base = api_base
    except Exception:
        pass

    return {"api_key": api_key, "api_base": api_base}
