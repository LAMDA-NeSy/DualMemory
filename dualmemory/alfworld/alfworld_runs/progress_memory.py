import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import openai

from api_config import apply_api_config, get_api_model
from utils import Model, get_chat

try:
    # Optional: use the shared task->item parser so milestone evidence is grounded to the correct object type.
    from stateinfo_transform.state_info_transform import extract_target_item as _extract_target_item
except Exception:
    _extract_target_item = None


VALID_ACTION_PREFIXES = (
    "go to",
    "open",
    "close",
    "take",
    "put",
    "move",
    "clean",
    "heat",
    "cool",
    "use",
    "look",
    "examine",
    "inventory",
)

ALFWORLD_TASK_TYPES = ("put", "clean", "heat", "cool", "examine", "puttwo")

_MILESTONE_KIND_FIND_PICK = "find_pick"
_MILESTONE_KIND_PUT = "put"
_MILESTONE_KIND_CLEAN = "clean"
_MILESTONE_KIND_HEAT = "heat"
_MILESTONE_KIND_COOL = "cool"
_MILESTONE_KIND_GOTO = "goto"
_MILESTONE_KIND_USE = "use"
_MILESTONE_KIND_OPEN = "open"

_ORDINAL_WORDS = {
    "first": 1,
    "1st": 1,
    "one": 1,
    "second": 2,
    "2nd": 2,
    "two": 2,
    "third": 3,
    "3rd": 3,
    "three": 3,
}


def infer_alfworld_task_type(task_text: str) -> Optional[str]:
    """Infer ALFWorld task type from the natural-language task string.

    ALFWorld tasks are typically grouped into 6 categories:
    put / clean / heat / cool / examine / puttwo.
    """
    if not task_text:
        return None

    text = task_text.strip().lower()

    # Strip common wrappers like "Task: ..." / "Your task is to: ..."
    m = re.search(r"(?:your\s+task\s+is\s+to|task)\s*:\s*(.*)", text)
    if m:
        text = m.group(1).strip()

    # Disambiguation: puttwo must be detected before put.
    # ALFWorld "pick_two_obj" tasks are commonly phrased with quantity words.
    if re.search(r"\btwo\b", text) or re.search(r"\bboth\b", text):
        return "puttwo"
    if re.search(r"\bclean\b", text):
        return "clean"
    if re.search(r"\bheat\b", text):
        return "heat"
    if re.search(r"\bcool\b", text):
        return "cool"
    if re.search(r"\bexamine\b", text) or re.search(r"\blook\s+at\b", text):
        return "examine"
    if re.search(r"\bput\b", text):
        return "put"

    return None


DEFAULT_EMBEDDING_MODEL = get_api_model("embedding", "text-embedding-3-small")
DEFAULT_PROGRESS_MEMORY_MODEL = get_api_model("progress_memory_planner", "gpt-4o-mini")

_SENTENCE_TRANSFORMER_MODELS = {"all-mpnet-base-v2"}
_ST_MODEL_CACHE: Dict[str, Any] = {}


def _is_sentence_transformer_model(model: str) -> bool:
    return (
        model in _SENTENCE_TRANSFORMER_MODELS
        or model.startswith("sentence-transformers/")
        or os.path.isdir(model)
    )


def _sentence_transformer_name(model: str) -> str:
    if model.startswith("sentence-transformers/"):
        return model.split("/", 1)[1]
    return model


def _sentence_transformer_cache_path(model_name: str) -> str:
    st_cache_root = os.path.expanduser("~/.cache/torch/sentence_transformers")
    return os.path.join(st_cache_root, f"sentence-transformers_{model_name}")


def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            f"SentenceTransformers is required for embedding model {model_name}. "
            "Install sentence-transformers and torch."
        ) from exc
    model = _ST_MODEL_CACHE.get(model_name)
    if model is None:
        if os.path.isdir(model_name):
            model = SentenceTransformer(model_name)
        else:
            local_path = _sentence_transformer_cache_path(model_name)
            has_weights = any(
                os.path.exists(os.path.join(local_path, fname))
                for fname in ("model.safetensors", "pytorch_model.bin")
            )

            # Prefer local cache when available to avoid any HuggingFace Hub calls (which can
            # time out even if most files are cached).
            if os.path.isdir(local_path) and has_weights:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                model = SentenceTransformer(local_path)
            else:
                try:
                    model = SentenceTransformer(model_name)
                except Exception:
                    # Some environments have an incomplete HF hub snapshot but a complete
                    # sentence-transformers cache (common if a download was interrupted).
                    if os.path.isdir(local_path) and has_weights:
                        os.environ.setdefault("HF_HUB_OFFLINE", "1")
                        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                        model = SentenceTransformer(local_path)
                    else:
                        raise
        _ST_MODEL_CACHE[model_name] = model
    return model


def _embed_texts_sentence_transformers(
    texts: List[str], model: str, batch_size: int
) -> List[List[float]]:
    if not texts:
        return []
    model_name = _sentence_transformer_name(model)
    st_model = _load_sentence_transformer(model_name)
    embeddings = st_model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return [list(map(float, emb)) for emb in embeddings]


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _embed_texts(texts: List[str], model: str, batch_size: int = 96) -> List[List[float]]:
    if _is_sentence_transformer_model(model):
        return _embed_texts_sentence_transformers(texts, model, batch_size)
    apply_api_config()
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = openai.Embedding.create(model=model, input=batch)
        data = sorted(response["data"], key=lambda item: item["index"])
        embeddings.extend([item["embedding"] for item in data])
    return embeddings


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _load_prompt(prompt_dir: str, filename: str, fallback: str = "") -> str:
    path = os.path.join(prompt_dir, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return fallback


def _extract_json_block(text: str) -> str:
    if "[" in text and "]" in text:
        return text[text.index("[") : text.rindex("]") + 1]
    if "{" in text and "}" in text:
        return text[text.index("{") : text.rindex("}") + 1]
    return text


def _safe_json_loads(text: str) -> Any:
    text = text.strip()
    text = _extract_json_block(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to remove trailing commas
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return json.loads(text)


def format_milestone_guide(milestones: List[str]) -> str:
    return "\n".join(f"{i + 1}. {m}" for i, m in enumerate(milestones))


_ALFWORLD_LOCATION_TYPES = {
    "cabinet",
    "countertop",
    "diningtable",
    "drawer",
    "fridge",
    "garbagecan",
    "microwave",
    "shelf",
    "sinkbasin",
    "stoveburner",
    "toaster",
    "coffeemachine",
    "sidetable",
    "bed",
    "desk",
    "dresser",
    "safe",
    "sofa",
    "coffeetable",
    "bathtubbasin",
    "handtowelholder",
    "towelholder",
    "toilet",
    "toiletpaperhanger",
}


def sanitize_milestone_guide(task_text: str, milestone_guide: List[str]) -> List[str]:
    """Remove brittle demo-specific instance IDs from milestones (e.g., 'cabinet 5').

    Unless the TASK explicitly mentions an exact instance (e.g., 'cabinet 5'), milestones should
    refer to location types (cabinet/drawer/...) rather than a specific numbered instance. This
    prevents the LLM from copying numbers from retrieved examples that may not exist in the
    current environment.
    """
    if not milestone_guide:
        return []
    task_low = str(task_text or "").lower()

    allow_numbered: set[str] = set()
    for loc in _ALFWORLD_LOCATION_TYPES:
        if re.search(rf"\b{re.escape(loc)}\s*(?:#|-)?\s*\d+\b", task_low):
            allow_numbered.add(loc)

    sanitized: List[str] = []
    for m in milestone_guide:
        text = str(m or "").strip()
        if not text:
            continue
        for loc in _ALFWORLD_LOCATION_TYPES:
            if loc in allow_numbered:
                continue
            text = re.sub(
                rf"\b{re.escape(loc)}\s*(?:#|-)?\s*\d+\b",
                loc,
                text,
                flags=re.IGNORECASE,
            )
        text = re.sub(r"\s+", " ", text).strip()
        sanitized.append(text)
    return sanitized


def extract_available_locations_from_trajectory(trajectory: List[Dict[str, str]]) -> List[str]:
    """Extract navigable targets like 'cabinet 1' from observation text.

    This is used to ground hint/action generation and avoid copying demo-specific targets.
    """
    known: set[str] = set()
    for step in trajectory or []:
        obs = str(step.get("observation", "") or "")
        for typ, idx in re.findall(r"\b([A-Za-z]+)\s+(\d+)\b", obs):
            typ_l = typ.lower()
            if typ_l in _ALFWORLD_LOCATION_TYPES:
                known.add(f"{typ_l} {idx}")
    return sorted(known, key=lambda s: (s.split()[0], int(s.split()[1])))


def format_available_locations(locations: List[str], max_items: int = 80) -> str:
    if not locations:
        return "- None"
    if max_items > 0:
        locations = locations[:max_items]
    return ", ".join(locations)


def extract_visited_locations_from_trajectory(trajectory: List[Dict[str, str]]) -> List[str]:
    visited: List[str] = []
    for step in trajectory or []:
        action = str(step.get("action", "") or "").strip()
        m = re.match(r"^go to\s+([A-Za-z]+\s+\d+)\s*$", action, re.IGNORECASE)
        if m:
            visited.append(m.group(1).lower())
    # keep order, unique
    seen = set()
    uniq: List[str] = []
    for loc in visited:
        if loc in seen:
            continue
        seen.add(loc)
        uniq.append(loc)
    return uniq


def format_visited_locations(locations: List[str], max_items: int = 30) -> str:
    if not locations:
        return "- None"
    if max_items > 0:
        locations = locations[:max_items]
    return ", ".join(locations)


def _mask_entity_ids(text: str) -> str:
    """Delexicalize numeric IDs (e.g., 'cabinet 6' -> 'cabinet [id]') for robustness."""
    if not text:
        return text
    return re.sub(r"\b([A-Za-z]+)\s+(\d+)\b", r"\1 [id]", text)


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", (text or "").strip().lower())


def _canonicalize_entity_phrase(phrase: str) -> str:
    """Canonicalize a possibly multi-word object/location phrase into ALFWorld-style token.

    Examples:
      - "garbage can" -> "garbagecan"
      - "soap bottle" -> "soapbottle"
      - "hand towel"  -> "handtowel"
    """
    phrase = (phrase or "").strip().lower()
    phrase = phrase.replace("-", " ")
    phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase)
    phrase = re.sub(r"\s+", " ", phrase)
    phrase = phrase.replace(" ", "")
    return _normalize_token(phrase)

def _extract_milestone_ordinal(milestone_text: str) -> int:
    """Extract ordinal index (1-based) from milestone text, defaulting to 1."""
    low = (milestone_text or "").lower()
    for word, idx in _ORDINAL_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", low):
            return idx
    return 1


def _entity_phrase_to_token_and_norm(entity_phrase: str) -> Tuple[str, str]:
    """Return (entity_token, entity_norm) for strings like 'cabinet 3' or 'pan 1'."""
    phrase = (entity_phrase or "").strip().lower()
    m = re.match(r"^([a-z]+)\s+(\d+)$", phrase)
    if m:
        token = m.group(1)
        return token, f"{token}{m.group(2)}"
    token = _canonicalize_entity_phrase(phrase)
    return token, _normalize_token(phrase)


def _obs_is_failure(observation: str) -> bool:
    return "nothing happens" in (observation or "").strip().lower()


def _extract_visible_items_from_observation(observation: str) -> List[Tuple[str, str]]:
    """Extract visible item instances from an observation.

    Returns list of (item_token, item_norm) where item_norm is like 'pan1'.
    """
    obs = (observation or "")
    found: List[Tuple[str, str]] = []

    # Common patterns:
    # - "On the desk 1, you see a ... , a desklamp 1, ..."
    # - "In it, you see a cd 3, a keychain 1, ..."
    for m in re.finditer(r"\byou\s+see\s+(.*)", obs, flags=re.IGNORECASE):
        chunk = m.group(1)
        for typ, idx in re.findall(r"\b([A-Za-z]+)\s+(\d+)\b", chunk):
            token = typ.lower()
            if token in _ALFWORLD_LOCATION_TYPES:
                continue
            found.append((token, f"{token}{idx}"))
    return found


def _extract_location_from_observation(observation: str) -> Optional[str]:
    obs = (observation or "").strip()
    # Prefer explicit arrivals: "You arrive at X."
    m = re.search(r"\byou\s+arrive\s+at\s+([A-Za-z]+\s+\d+)\b", obs, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Fallback: "On the X, you see ..."
    m = re.search(r"\bon\s+the\s+([A-Za-z]+\s+\d+)\b", obs, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Fallback: "The drawer 2 is closed/open."
    m = re.search(r"\bthe\s+([A-Za-z]+\s+\d+)\s+is\s+(?:closed|open)\b", obs, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


class TrajectoryState:
    """Explicit state derived from (action, observation) trajectory steps."""

    def __init__(self) -> None:
        self.current_location: Optional[str] = None  # phrase like 'drawer 2'
        self.current_location_norm: Optional[str] = None  # like 'drawer2'
        self.location_status: Dict[str, str] = {}  # norm -> 'open'|'closed'|'unknown'
        self.location_phrase_by_norm: Dict[str, str] = {}  # norm -> phrase like 'drawer 2'

        self.visited_locations: set[str] = set()  # norms
        self.opened_locations: set[str] = set()  # norms

        self.held_item: Optional[str] = None  # norm like 'pan1'
        self.held_item_token: Optional[str] = None  # 'pan'

        self.picked_order_by_token: Dict[str, List[str]] = defaultdict(list)  # token -> [item_norm]
        self.item_properties: Dict[str, set[str]] = defaultdict(set)  # item_norm -> {'cleaned','heated','cooled','used'}
        self.put_events: List[Dict[str, str]] = []  # {item_norm,item_token,target_norm,target_token}

        self.visible_items_by_location: Dict[str, set[str]] = defaultdict(set)  # loc_norm -> {item_token}
        self.last_observation_by_location: Dict[str, str] = {}  # loc_norm -> raw observation

    @staticmethod
    def from_trajectory(trajectory: List[Dict[str, str]]) -> "TrajectoryState":
        state = TrajectoryState()
        for step in trajectory or []:
            state.update(str(step.get("action", "") or ""), str(step.get("observation", "") or ""))
        return state

    def update(self, action: str, observation: str) -> None:
        act = (action or "").strip()
        low = act.lower()
        obs = (observation or "").strip()
        obs_low = obs.lower()

        if low.startswith("go to"):
            m = re.match(r"^go to\s+([A-Za-z]+\s+\d+)\s*$", act, flags=re.IGNORECASE)
            if m and not _obs_is_failure(obs):
                loc_phrase = m.group(1).lower()
                _, loc_norm = _entity_phrase_to_token_and_norm(loc_phrase)
                self.current_location = loc_phrase
                self.current_location_norm = loc_norm
                self.location_phrase_by_norm[loc_norm] = loc_phrase
                self.visited_locations.add(loc_norm)
                if " is closed" in obs_low:
                    self.location_status[loc_norm] = "closed"
                elif " is open" in obs_low:
                    self.location_status[loc_norm] = "open"
                else:
                    self.location_status.setdefault(loc_norm, "unknown")

        elif low.startswith("open"):
            m = re.match(r"^open\s+([A-Za-z]+\s+\d+)\s*$", act, flags=re.IGNORECASE)
            if m and not _obs_is_failure(obs):
                loc_phrase = m.group(1).lower()
                _, loc_norm = _entity_phrase_to_token_and_norm(loc_phrase)
                self.location_status[loc_norm] = "open"
                self.opened_locations.add(loc_norm)
                self.current_location = loc_phrase
                self.current_location_norm = loc_norm
                self.location_phrase_by_norm[loc_norm] = loc_phrase

        elif low.startswith("take"):
            # e.g., "take apple 3 from garbagecan 1"
            m = re.match(r"^take\s+([A-Za-z]+\s+\d+)\s+from\s+([A-Za-z]+\s+\d+)\s*$", act, flags=re.IGNORECASE)
            if m and not _obs_is_failure(obs) and "you pick up the" in obs_low:
                item_phrase = m.group(1).lower()
                item_token, item_norm = _entity_phrase_to_token_and_norm(item_phrase)
                self.held_item = item_norm
                self.held_item_token = item_token
                if item_norm not in self.picked_order_by_token[item_token]:
                    self.picked_order_by_token[item_token].append(item_norm)

        elif low.startswith("put") or low.startswith("move"):
            # e.g., "put apple 3 in/on sidetable 1" / "move the apple 3 to the sidetable 1"
            m = re.match(
                r"^(?:put|move)\s+([A-Za-z]+\s+\d+)\s+(?:in/on|into|in|on|to)\s+([A-Za-z]+\s+\d+)\s*$",
                act,
                flags=re.IGNORECASE,
            )
            if m and not _obs_is_failure(obs) and re.search(r"\byou\s+(?:put|move)\s+the\b", obs_low):
                item_phrase = m.group(1).lower()
                target_phrase = m.group(2).lower()
                item_token, item_norm = _entity_phrase_to_token_and_norm(item_phrase)
                target_token, target_norm = _entity_phrase_to_token_and_norm(target_phrase)
                self.put_events.append(
                    {
                        "item_norm": item_norm,
                        "item_token": item_token,
                        "target_norm": target_norm,
                        "target_token": target_token,
                    }
                )
                # One-hand assumption: if the item we placed is the one we held, clear.
                if self.held_item == item_norm:
                    self.held_item = None
                    self.held_item_token = None

        elif low.startswith("clean") or low.startswith("heat") or low.startswith("cool"):
            m = re.match(
                r"^(clean|heat|cool)\s+([A-Za-z]+\s+\d+)\s+with\s+([A-Za-z]+\s+\d+)\s*$",
                act,
                flags=re.IGNORECASE,
            )
            if m and not _obs_is_failure(obs):
                verb = m.group(1).lower()
                item_phrase = m.group(2).lower()
                item_token, item_norm = _entity_phrase_to_token_and_norm(item_phrase)
                # Evidence must appear in simulator wording.
                if re.search(rf"\byou\s+{re.escape(verb)}\s+the\s+{re.escape(item_token)}\b", obs_low):
                    if verb == "clean":
                        self.item_properties[item_norm].add("cleaned")
                    elif verb == "heat":
                        self.item_properties[item_norm].add("heated")
                    elif verb == "cool":
                        self.item_properties[item_norm].add("cooled")

        elif low.startswith("use"):
            # "use desklamp 1"
            m = re.match(r"^use\s+([A-Za-z]+\s+\d+)\s*$", act, flags=re.IGNORECASE)
            if m and not _obs_is_failure(obs):
                item_phrase = m.group(1).lower()
                item_token, item_norm = _entity_phrase_to_token_and_norm(item_phrase)
                if re.search(rf"\byou\s+turn\s+on\s+the\s+{re.escape(item_token)}\b", obs_low):
                    self.item_properties[item_norm].add("used")

        # Update current location from observation text when available.
        loc_phrase = _extract_location_from_observation(obs)
        if loc_phrase and not _obs_is_failure(obs):
            _, loc_norm = _entity_phrase_to_token_and_norm(loc_phrase)
            self.current_location = loc_phrase
            self.current_location_norm = loc_norm
            self.location_phrase_by_norm[loc_norm] = loc_phrase
            self.visited_locations.add(loc_norm)
            self.last_observation_by_location[loc_norm] = obs

            for item_token, _item_norm in _extract_visible_items_from_observation(obs):
                self.visible_items_by_location[loc_norm].add(item_token)


def _matches_location_target(target_norm: str, location_norm: Optional[str]) -> bool:
    if not target_norm or not location_norm:
        return False
    # If target includes a specific ID (e.g., cabinet3), require exact match.
    if re.search(r"\d", target_norm):
        return location_norm == target_norm
    # Otherwise allow any instance (cabinet1/cabinet2...).
    return location_norm.startswith(target_norm)


def _matches_put_target(target_norm: Optional[str], put_event: Dict[str, str]) -> bool:
    if not target_norm:
        return True
    # If target_norm includes id, match exact; else prefix match.
    return _matches_location_target(target_norm, put_event.get("target_norm"))


def _evidence_step_index(recent_steps: List[Dict[str, str]], predicate) -> int:
    for i in range(len(recent_steps), 0, -1):
        step = recent_steps[i - 1]
        if predicate(str(step.get("action", "") or ""), str(step.get("observation", "") or "")):
            return i
    return 0


def parse_milestone_signature(milestone_text: str) -> Dict[str, Optional[str]]:
    """Parse a milestone into a coarse signature for retrieval filtering.

    Returns keys:
      - kind: one of *_MILESTONE_KIND_*
      - object: canonical object token (lowercase), if any
      - target: canonical target/location token (lowercase), if any

    This is intentionally simple and avoids semantic similarity. It prevents obviously wrong
    demos (e.g., soapbar) from being retrieved for a soapbottle milestone.
    输入一句自然语言（如 "Put the clean apple in the fridge"），输出一个字典，提取出动作类型（kind）、操作对象（object）和目标容器（target）
    """
    text = (milestone_text or "").strip()
    low = text.lower()

    # Dispose/throw away milestones: treat as a PUT-like kind so they don't pollute FIND_PICK retrieval.
    if re.search(r"\bdispose\b|\bthrow\s+away\b|\btrash\b", low):
        m = re.search(
            r"\b(?:dispose\s+of|throw\s+away|trash)\b\s+(?:the\s+)?(?:first|second|third)?\s*([a-z0-9_]+(?:\s+[a-z0-9_]+)*)",
            low,
            flags=re.IGNORECASE,
        )
        obj = _canonicalize_entity_phrase(m.group(1)) if m else None
        tgt = "garbagecan" if re.search(r"\bgarbage\s*can\b|\bgarbagecan\b|\btrashcan\b", low) else None
        return {"kind": _MILESTONE_KIND_PUT, "object": obj, "target": tgt}

    # retrieve ... (often used as a synonym for "find and take")
    # Handle "... from ..." first so object doesn't swallow the location phrase.
    m = re.search(
        r"\bretrieve\b\s+(?:the\s+)?(?:first|second|third)?\s*([a-z0-9_]+(?:\s+[a-z0-9_]+)*)\s+"
        r"from\s+(?:the\s+)?([a-z0-9_]+(?:\s+[a-z0-9_]+)*)",
        low,
        flags=re.IGNORECASE,
    )
    if m:
        obj = _canonicalize_entity_phrase(m.group(1))
        tgt = _canonicalize_entity_phrase(m.group(2))
        return {"kind": _MILESTONE_KIND_FIND_PICK, "object": obj, "target": tgt}
    m = re.search(
        r"\bretrieve\b\s+(?:the\s+)?(?:first|second|third)?\s*([a-z0-9_]+(?:\s+[a-z0-9_]+)*)",
        low,
        flags=re.IGNORECASE,
    )
    if m:
        obj = _canonicalize_entity_phrase(m.group(1))
        return {"kind": _MILESTONE_KIND_FIND_PICK, "object": obj, "target": None}

    # locate/search for ... (often used as a synonym for "find")
    m = re.search(
        r"\b(?:locate|search\s+for)\b\s+(?:the\s+|a\s+|an\s+)?(?:first|second|third)?\s*([a-z0-9_]+(?:\s+[a-z0-9_]+)*)",
        low,
        flags=re.IGNORECASE,
    )
    if m:
        obj = _canonicalize_entity_phrase(m.group(1))
        if obj in {"", "and", "then", "to"}:
            return {"kind": None, "object": None, "target": None}
        return {"kind": _MILESTONE_KIND_FIND_PICK, "object": obj, "target": None}

    # find & pick up / find & take / find & collect (synonyms appear in some libraries)
    m = re.search(
        r"\bfind\b[\s\S]*?\b(pick\s+up|take|collect|get|grab|retrieve)\b\s+"
        r"(?:the\s+)?(?:first|second|third)?\s*([a-z0-9_]+(?:\s+[a-z0-9_]+)*)",
        low,
        flags=re.IGNORECASE,
    )
    if m:
        return {"kind": _MILESTONE_KIND_FIND_PICK, "object": _canonicalize_entity_phrase(m.group(2)), "target": None}

    # "Pick up/Take/Collect/Get/Grab/Retrieve the X" (without explicit "find")
    m = re.search(
        r"\b(pick\s+up|take|collect|get|grab|retrieve)\b\s+(?:the\s+)?(?:first|second|third)?\s*([a-z0-9_]+(?:\s+[a-z0-9_]+)*)",
        low,
        flags=re.IGNORECASE,
    )
    if m:
        return {"kind": _MILESTONE_KIND_FIND_PICK, "object": _canonicalize_entity_phrase(m.group(2)), "target": None}

    # Some libraries phrase the milestone as "Find the X" (without explicit pick-up).
    m = re.search(
        r"\bfind\b\s+(?:the\s+|a\s+|an\s+)?(?:first|second|third)?\s*([a-z0-9_]+(?:\s+[a-z0-9_]+)*)",
        low,
        flags=re.IGNORECASE,
    )
    if m:
        obj = _canonicalize_entity_phrase(m.group(1))
        # Guard against "find and <verb> ..." accidentally binding 'and' as the object.
        if obj in {
            "",
            "and",
            "then",
            "to",
            "a",
            "an",
            "the",
            "collect",
            "get",
            "grab",
            "retrieve",
            "take",
            "pick",
            "up",
        }:
            return {"kind": None, "object": None, "target": None}
        return {"kind": _MILESTONE_KIND_FIND_PICK, "object": obj, "target": None}

    # clean / heat / cool
    for kind, kw in (
        (_MILESTONE_KIND_CLEAN, "clean"),
        (_MILESTONE_KIND_HEAT, "heat"),
        (_MILESTONE_KIND_COOL, "cool"),
    ):
        m = re.search(
            rf"\b{kw}\b\s+(?:the\s+)?([a-z0-9_]+(?:\s+[a-z0-9_]+)*)",
            low,
            flags=re.IGNORECASE,
        )
        if m:
            return {"kind": kind, "object": _canonicalize_entity_phrase(m.group(1)), "target": None}

    # go to ...
    m = re.search(
        r"\bgo\s+to\b\s+(?:the\s+)?([a-z0-9_]+(?:\s+[a-z0-9_]+)*)",
        low,
        flags=re.IGNORECASE,
    )
    if m:
        return {"kind": _MILESTONE_KIND_GOTO, "object": None, "target": _canonicalize_entity_phrase(m.group(1))}

    # open ...
    m = re.search(
        r"\bopen\b\s+(?:the\s+)?([a-z0-9_]+(?:\s+[a-z0-9_]+)*)",
        low,
        flags=re.IGNORECASE,
    )
    if m:
        return {"kind": _MILESTONE_KIND_OPEN, "object": None, "target": _canonicalize_entity_phrase(m.group(1))}

    # put/place/move X (in/on/into/to) Y
    m = re.search(
        r"\b(put|place|move)\b\s+(?:the\s+)?(?:first|second|third)?\s*([a-z0-9_]+(?:\s+[a-z0-9_]+)*)\s+"
        r"(?:in/on|into|in|on|to|near)\s+(?:the\s+)?([a-z0-9_]+(?:\s+[a-z0-9_]+)*)",
        low,
        flags=re.IGNORECASE,
    )
    if m:
        return {
            "kind": _MILESTONE_KIND_PUT,
            "object": _canonicalize_entity_phrase(m.group(2)),
            "target": _canonicalize_entity_phrase(m.group(3)),
        }

    # use ...
    m = re.search(
        r"\buse\b\s+(?:the\s+)?([a-z0-9_]+(?:\s+[a-z0-9_]+)*)",
        low,
        flags=re.IGNORECASE,
    )
    if m:
        return {"kind": _MILESTONE_KIND_USE, "object": _canonicalize_entity_phrase(m.group(1)), "target": None}

    return {"kind": None, "object": None, "target": None}


def _segment_has_evidence(
    segment: List[Dict[str, Any]],
    kind: Optional[str],
    obj: Optional[str],
    target: Optional[str],
) -> bool:
    if not segment or not kind:
        return False
    obj_n = _normalize_token(obj or "")
    tgt_n = _normalize_token(target or "")

    obs_text = "\n".join(str(s.get("observation", "") or "") for s in segment)
    low = obs_text.lower()

    if kind == _MILESTONE_KIND_FIND_PICK and obj_n:
        return re.search(rf"\byou\s+pick\s+up\s+the\s+{re.escape(obj_n)}\b", low) is not None
    if kind == _MILESTONE_KIND_CLEAN and obj_n:
        return re.search(rf"\byou\s+clean\s+the\s+{re.escape(obj_n)}\b", low) is not None
    if kind == _MILESTONE_KIND_HEAT and obj_n:
        return re.search(rf"\byou\s+heat\s+the\s+{re.escape(obj_n)}\b", low) is not None
    if kind == _MILESTONE_KIND_COOL and obj_n:
        return re.search(rf"\byou\s+cool\s+the\s+{re.escape(obj_n)}\b", low) is not None
    if kind == _MILESTONE_KIND_PUT and obj_n:
        return re.search(rf"\byou\s+(?:put|move)\s+the\s+{re.escape(obj_n)}\b", low) is not None
    if kind == _MILESTONE_KIND_GOTO and tgt_n:
        return re.search(rf"\byou\s+arrive\s+at\s+{re.escape(tgt_n)}\b", low) is not None
    if kind == _MILESTONE_KIND_OPEN and tgt_n:
        return re.search(rf"\bthe\s+{re.escape(tgt_n)}\s+.*\bis\s+open\b", low) is not None
    if kind == _MILESTONE_KIND_USE and obj_n:
        return re.search(rf"\byou\s+turn\s+on\s+the\s+{re.escape(obj_n)}\b", low) is not None
    return False


def _bind_demo_step_to_query(
    step: Dict[str, Any],
    query_kind: Optional[str],
    query_object: Optional[str],
    query_target: Optional[str],
    source_object: Optional[str] = None,
    source_target: Optional[str] = None,
) -> Dict[str, str]:
    """Bind a retrieved demo step to the current milestone signature.

    This is a *mechanism* fix: when the library lacks exact object coverage (e.g., no 'soapbottle'
    milestones), we still want demos to illustrate the same subtask pattern without misleading the
    model with a different object name (e.g., 'soapbar').
    """
    action = str(step.get("action", "") or "")
    observation = str(step.get("observation", "") or "")

    obj = _normalize_token(query_object or "")
    tgt = _normalize_token(query_target or "")
    src_obj = _normalize_token(source_object or "")
    src_tgt = _normalize_token(source_target or "")

    allowed_obj_verbs: set[str] = set()
    if query_kind == _MILESTONE_KIND_FIND_PICK:
        allowed_obj_verbs = {"take"}
    elif query_kind == _MILESTONE_KIND_CLEAN:
        allowed_obj_verbs = {"clean"}
    elif query_kind == _MILESTONE_KIND_HEAT:
        allowed_obj_verbs = {"heat"}
    elif query_kind == _MILESTONE_KIND_COOL:
        allowed_obj_verbs = {"cool"}
    elif query_kind == _MILESTONE_KIND_PUT:
        allowed_obj_verbs = {"put", "move"}

    # First, do a conservative token swap for the demo's own object/target tokens (when known).
    # This reduces contradictions like: "you see soapbar" -> "take soapbottle".
    if obj and src_obj and src_obj != obj:
        action = re.sub(rf"\b{re.escape(src_obj)}\b", obj, action, flags=re.IGNORECASE)
        observation = re.sub(rf"\b{re.escape(src_obj)}\b", obj, observation, flags=re.IGNORECASE)
    if tgt and src_tgt and src_tgt != tgt:
        action = re.sub(rf"\b{re.escape(src_tgt)}\b", tgt, action, flags=re.IGNORECASE)
        observation = re.sub(rf"\b{re.escape(src_tgt)}\b", tgt, observation, flags=re.IGNORECASE)

    if obj and allowed_obj_verbs:
        # Action object slot binding for relevant verbs only (avoid corrupting unrelated segments).
        def _swap_obj(m: re.Match[str]) -> str:
            verb = m.group(1)
            suffix = m.group(2) or ""
            return f"{verb} {obj}{suffix}"

        for verb in sorted(allowed_obj_verbs):
            action = re.sub(
                rf"^({re.escape(verb)})\s+[a-z0-9_]+(\s+\d+)?\b",
                _swap_obj,
                action,
                flags=re.IGNORECASE,
            )

        # Observation evidence phrase binding (keep it narrow to avoid rewriting unrelated text).
        if query_kind == _MILESTONE_KIND_FIND_PICK:
            observation = re.sub(
                r"\bYou\s+pick\s+up\s+the\s+[a-z0-9_]+(\s+\d+)?\b",
                lambda m: f"You pick up the {obj}{m.group(1) or ''}",
                observation,
                flags=re.IGNORECASE,
            )
        elif query_kind == _MILESTONE_KIND_CLEAN:
            observation = re.sub(
                r"\bYou\s+clean\s+the\s+[a-z0-9_]+(\s+\d+)?\b",
                lambda m: f"You clean the {obj}{m.group(1) or ''}",
                observation,
                flags=re.IGNORECASE,
            )
        elif query_kind == _MILESTONE_KIND_HEAT:
            observation = re.sub(
                r"\bYou\s+heat\s+the\s+[a-z0-9_]+(\s+\d+)?\b",
                lambda m: f"You heat the {obj}{m.group(1) or ''}",
                observation,
                flags=re.IGNORECASE,
            )
        elif query_kind == _MILESTONE_KIND_COOL:
            observation = re.sub(
                r"\bYou\s+cool\s+the\s+[a-z0-9_]+(\s+\d+)?\b",
                lambda m: f"You cool the {obj}{m.group(1) or ''}",
                observation,
                flags=re.IGNORECASE,
            )
        elif query_kind == _MILESTONE_KIND_PUT:
            observation = re.sub(
                r"\bYou\s+(?:put|move)\s+the\s+[a-z0-9_]+(\s+\d+)?\b",
                lambda m: f"You move the {obj}{m.group(1) or ''}",
                observation,
                flags=re.IGNORECASE,
            )

    if tgt:
        # Action target slot binding for navigation/open.
        def _swap_tgt(m: re.Match[str]) -> str:
            verb = m.group(1)
            suffix = m.group(2) or ""
            return f"{verb} {tgt}{suffix}"

        action = re.sub(r"^(go to)\s+[a-z0-9_]+(\s+\d+)?\b", _swap_tgt, action, flags=re.IGNORECASE)
        action = re.sub(r"^(open)\s+[a-z0-9_]+(\s+\d+)?\b", _swap_tgt, action, flags=re.IGNORECASE)

        # Observation: keep only arrival binding (most useful for grounding).
        observation = re.sub(
            r"\bYou\s+arrive\s+at\s+[a-z0-9_]+(\s+\d+)?\b",
            lambda m: f"You arrive at {tgt}{m.group(1) or ''}",
            observation,
            flags=re.IGNORECASE,
        )

    return {"action": action, "observation": observation}


def format_trajectory_steps(trajectory: List[Dict[str, str]], max_steps: int = 8) -> str:
    if max_steps > 0:
        trajectory = trajectory[-max_steps:]
    parts = []
    for step in trajectory:
        parts.append(f"Action: {step['action']}\nObservation: {step['observation']}")
    return "\n".join(parts) if parts else "- None"


def format_segments(
    segments: List[List[Dict[str, str]]],
    max_segment_steps: int = 6,
    mask_ids: bool = False,
) -> str:
    formatted = []
    for idx, segment in enumerate(segments, start=1):
        limited = segment[:max_segment_steps] if max_segment_steps > 0 else segment
        if mask_ids:
            limited = [
                {
                    "action": _mask_entity_ids(str(step.get("action", ""))),
                    "observation": _mask_entity_ids(str(step.get("observation", ""))),
                }
                for step in limited
            ]
        formatted.append(f"Demonstration {idx}:\n{format_trajectory_steps(limited, max_steps=0)}")
    return "\n\n".join(formatted) if formatted else "- None"


def format_task_level_demonstrations(
    tasks: List[Dict[str, Any]],
    *,
    max_demos: int = 2,
    max_steps: int = 12,
    mask_ids: bool = True,
) -> str:
    """Format retrieved similar tasks as task-level demonstrations for prompting.

    Each demo includes the example task text, its milestone guide, and a truncated trajectory.
    """
    if not tasks:
        return "- None"

    blocks: List[str] = []
    for idx, task in enumerate(tasks[: max(0, int(max_demos))] or [], start=1):
        task_text = str(task.get("task", "") or "").strip()
        guide = task.get("milestone_guide", []) or []
        traj = task.get("trajectory", []) or []

        if mask_ids:
            task_text = _mask_entity_ids(task_text)
            masked_traj: List[Dict[str, str]] = []
            for step in traj[: max(0, int(max_steps))] if max_steps > 0 else traj:
                masked_traj.append(
                    {
                        "action": _mask_entity_ids(str(step.get("action", "") or "")),
                        "observation": _mask_entity_ids(str(step.get("observation", "") or "")),
                    }
                )
            traj_text = format_trajectory_steps(masked_traj, max_steps=0)
        else:
            limited = traj[: max(0, int(max_steps))] if max_steps > 0 else traj
            traj_text = format_trajectory_steps(limited, max_steps=0)

        guide_text = format_milestone_guide(list(guide)) if guide else "- None"
        blocks.append(
            "\n".join(
                [
                    f"Demonstration {idx}:",
                    f"Task: {task_text or '- None'}",
                    "Milestone action guide:",
                    guide_text,
                    "Trajectory (patterns only; do NOT copy IDs/targets):",
                    traj_text or "- None",
                ]
            )
        )

    return "\n\n".join(blocks) if blocks else "- None"


def _is_valid_action(action: str) -> bool:
    action = action.lower().strip()
    return any(action.startswith(prefix) for prefix in VALID_ACTION_PREFIXES)


def parse_env_history(env_history: str) -> Optional[Dict[str, Any]]:
    
    task_marker = "Here is the task:"
    task_idx = env_history.find(task_marker)
    if task_idx == -1:
        task_marker = "Here is the task."
        task_idx = env_history.find(task_marker)
    if task_idx == -1:
        return None

    # 获取标记之后的所有内容，这部分包含了具体的任务描述和后续的交互过程
    after_task = env_history[task_idx + len(task_marker) :].strip()
    lines = after_task.splitlines()

    task_lines: List[str] = []
    trajectory: List[Dict[str, str]] = []
    current_action: Optional[str] = None
    current_obs: List[str] = []

    for line in lines:
        # 如果以 > 开头，说明是动作
        if line.strip().startswith(">"):
            # 如果之前有动作，先保存上一个动作和观察
            if current_action is not None:
                observation = "\n".join(current_obs).strip()
                trajectory.append({"action": current_action, "observation": observation})
            # 更新当前动作
            current_action = line.strip()[1:].strip()
            current_obs = []
        else:
            if current_action is None:
                if line.strip():
                    task_lines.append(line.strip())
            else:
                current_obs.append(line)

    if current_action is not None:
        observation = "\n".join(current_obs).strip()
        trajectory.append({"action": current_action, "observation": observation})

    if not task_lines:
        return None

    task_text = _extract_task_line(task_lines)
    filtered_trajectory = [step for step in trajectory if _is_valid_action(step["action"])]

    return {
        "task": task_text,
        "trajectory": filtered_trajectory,
        "actions": [step["action"] for step in filtered_trajectory],
    }


def _extract_task_line(lines: List[str]) -> str:
    for line in lines:
        if "task" in line.lower() and ":" in line:
            return line.strip()
    for line in lines:
        if "task" in line.lower():
            return line.strip()
    return "\n".join(lines).strip()


def parse_trial_log(text: str) -> List[Dict[str, Any]]:
    pattern = re.compile(
        r"#####\s*Environment #(\d+):\s*(.*?)\s*STATUS:\s*(OK|FAIL)\s*#####",
        re.DOTALL,
    )
    results = []
    for match in pattern.finditer(text):
        env_id = int(match.group(1))
        env_history = match.group(2).strip()
        status = match.group(3).strip()
        results.append(
            {
                "env_id": env_id,
                "env_history": env_history,
                "status": status,
            }
        )
    return results


def extract_milestones(
    task: str,
    actions: List[str],
    model: Model,
    prompt_dir: str,
    max_tokens: int = 2048,
) -> List[Dict[str, Any]]:
    if not actions:
        return []

    trajectory_text = "\n".join(f"{i}. {a}" for i, a in enumerate(actions, start=1))



    prompt_template = _load_prompt(prompt_dir, "progress_memory_milestone_extract.txt")
    prompt = prompt_template.format(TASK=task, TRAJECTORY=trajectory_text)

    response = get_chat(prompt=prompt, model=model, temperature=0.0, max_tokens=max_tokens)
    try:
        parsed = _safe_json_loads(response)
    except Exception:
        return []

    if isinstance(parsed, dict) and "milestones" in parsed:
        parsed = parsed["milestones"]

    if not isinstance(parsed, list):
        return []

    milestones: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        milestone = str(item.get("milestone", "")).strip()
        action_ids = item.get("actions", [])
        if not milestone or not isinstance(action_ids, list):
            continue
        action_indices = _normalize_action_indices(action_ids, len(actions))
        if not action_indices:
            continue
        milestones.append({"milestone": milestone, "actions": action_indices})
    return milestones


def _normalize_action_indices(action_ids: List[Any], action_count: int) -> List[int]:
    indices: List[int] = []
    for idx in action_ids:
        try:
            indices.append(int(idx))
        except (TypeError, ValueError):
            continue
    if not indices:
        return []
    uses_zero = any(i == 0 for i in indices)
    if not uses_zero and min(indices) >= 1:
        indices = [i - 1 for i in indices]
    return [i for i in indices if 0 <= i < action_count]


def parse_milestone_guide(raw_text: str) -> List[str]:
    try:
        parsed = _safe_json_loads(raw_text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass

    lines = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d+\.", line):
            lines.append(re.sub(r"^\d+\.\s*", "", line))
        elif line.startswith("- "):
            lines.append(line[2:].strip())
    return lines


def parse_hint_milestone_index(hint: str, milestone_guide: List[str]) -> Optional[int]:
    match = re.search(r"Milestone\s*(\d+)", hint, re.IGNORECASE)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(milestone_guide):
            return idx
    for i, milestone in enumerate(milestone_guide):
        if milestone.lower() in hint.lower():
            return i
    return None


def refine_milestone_guide(
    task_text: str,
    milestone_guide: List[str],
    prompt_dir: str,
    model: Model,
    max_milestones: int = 6,
) -> List[str]:
    """
    LLM-based refinement for milestone guides to keep decomposition consistent and atomic.
    Keeps the paper setting: global guidance is produced by (progress memory library retrieval) + LLM.
    """
    if not milestone_guide:
        return []
    prompt_template = _load_prompt(prompt_dir=prompt_dir, filename="progress_memory_milestone_refine.txt")
    prompt = prompt_template.format(
        TASK=task_text,
        RAW_GUIDE=format_milestone_guide(milestone_guide),
    )
    response = get_chat(prompt=prompt, model=model, temperature=0.0, max_tokens=512)
    parsed = parse_milestone_guide(response)
    refined = parsed or milestone_guide

    # If the guide is still too long, ask the LLM once more to merge steps.
    # This stays within the paper setting: global guidance is produced by retrieval + LLM (no hand-written parsing).
    if max_milestones and len(refined) > max_milestones:
        compress_prompt = (
            prompt
            + f"\n\nYour previous rewrite is too long ({len(refined)} milestones). "
            f"Rewrite it again into <= {max_milestones} milestones by MERGING steps. "
            "Output ONLY a JSON array."
        )
        response2 = get_chat(prompt=compress_prompt, model=model, temperature=0.0, max_tokens=512)
        parsed2 = parse_milestone_guide(response2)
        if parsed2:
            refined = parsed2

    return sanitize_milestone_guide(task_text, refined)


class MilestoneLibrary:
    def __init__(self, library_path: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.library_path = library_path
        self.embedding_model = embedding_model
        self.tasks: List[Dict[str, Any]] = []
        self.milestones: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.library_path):
            return
        with open(self.library_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.tasks = data.get("tasks", [])
        self.embedding_model = data.get("embedding_model", self.embedding_model)

        for task in self.tasks:
            if not task.get("task_type"):
                task["task_type"] = infer_alfworld_task_type(task.get("task", ""))
            if "task_embedding" in task:
                task["task_embedding"] = _normalize(task["task_embedding"])

            for milestone in task.get("milestones", []):
                milestone_entry = dict(milestone)
                milestone_entry["task_id"] = task.get("task_id")
                milestone_entry["task_type"] = task.get("task_type")
                sig = parse_milestone_signature(str(milestone_entry.get("milestone", "") or ""))
                milestone_entry["milestone_kind"] = sig.get("kind")
                milestone_entry["milestone_object"] = sig.get("object")
                milestone_entry["milestone_target"] = sig.get("target")
                milestone_entry["milestone_embedding"] = _normalize(
                    milestone_entry.get("milestone_embedding", [])
                )
                self.milestones.append(milestone_entry)

    def has_data(self) -> bool:
        return bool(self.tasks) and bool(self.milestones)

    def embed_query(self, text: str) -> List[float]:
        embedding = _embed_texts([text], model=self.embedding_model)[0]
        return _normalize(embedding)

    def retrieve_similar_tasks(
        self,
        task_text: str,
        top_k: int = 2,
        task_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.tasks:
            return []
        target_type = task_type or infer_alfworld_task_type(task_text)
        candidates = (
            [t for t in self.tasks if t.get("task_type") == target_type]
            if target_type
            else list(self.tasks)
        )
        if not candidates:
            return []

        query = self.embed_query(task_text)
        scored = []
        for task in candidates:
            embedding = task.get("task_embedding")
            if not embedding:
                continue
            score = _cosine_similarity(query, embedding)
            trajectory_len = len(task.get("trajectory", []))
            scored.append((score, trajectory_len, task))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[:top_k]
        top.sort(key=lambda item: item[1])
        return [item[2] for item in top]

    def retrieve_similar_milestones(
        self,
        milestone_text: str,
        top_k: int = 2,
        exclude_task_ids: Optional[Iterable[str]] = None,
        task_type: Optional[str] = None,
        infer_type_if_missing: bool = True,
        distinct_task: bool = True,
        require_kind_match: bool = True,
        require_object_match: bool = True,
        require_target_match: bool = False,
        use_evidence_bonus: bool = True,
    ) -> List[Dict[str, Any]]:
        if not self.milestones:
            return []
        exclude = set(exclude_task_ids or [])
        # By default, we infer task_type from the milestone text to keep retrieval within the same
        # broad task category. Callers can disable this (infer_type_if_missing=False) to allow
        # cross-task-type milestone retrieval.
        if task_type is None:
            target_type = infer_alfworld_task_type(milestone_text) if infer_type_if_missing else None
        else:
            target_type = task_type
        query_sig = parse_milestone_signature(milestone_text)
        q_kind = query_sig.get("kind")
        q_obj = query_sig.get("object")
        q_tgt = query_sig.get("target")
        query = self.embed_query(milestone_text)
        scored = []
        for milestone in self.milestones:
            if milestone.get("task_id") in exclude:
                continue
            if target_type and milestone.get("task_type") != target_type:
                continue
            if require_kind_match and q_kind:
                cand_kind = milestone.get("milestone_kind")
                if not cand_kind or cand_kind != q_kind:
                    continue
            if require_object_match and q_obj:
                cand_obj = milestone.get("milestone_object")
                if not cand_obj or cand_obj != q_obj:
                    continue
            if require_target_match and q_tgt:
                cand_tgt = milestone.get("milestone_target")
                if not cand_tgt or cand_tgt != q_tgt:
                    continue
            embedding = milestone.get("milestone_embedding")
            if not embedding:
                continue
            score = _cosine_similarity(query, embedding)
            segment = milestone.get("segment") or []
            evidence_bonus = (
                0.05 if (use_evidence_bonus and _segment_has_evidence(segment, q_kind, q_obj, q_tgt)) else 0.0
            )
            scored.append((score + evidence_bonus, score, milestone))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

        if not distinct_task:
            return [item[2] for item in scored[:top_k]]

        selected = []
        used_tasks = set()
        for _, __, milestone in scored:
            task_id = milestone.get("task_id")
            if task_id in used_tasks:
                continue
            selected.append(milestone)
            used_tasks.add(task_id)
            if len(selected) >= top_k:
                break
        return selected


class ProgressMemoryPlanner:
    def __init__(
        self,
        library_path: str,
        prompt_dir: str,
        model_name: Model = DEFAULT_PROGRESS_MEMORY_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        top_tasks: int = 3,
        top_task_demos: int = 0,
        top_milestones: int = 3,
        max_history_steps: int = 0,
        max_segment_steps: int = 0,
        milestone_check_mode: str = "llm",
    ) -> None:
        self.library = MilestoneLibrary(library_path, embedding_model=embedding_model)
        self.prompt_dir = prompt_dir
        self.model_name = model_name
        self.top_tasks = top_tasks
        self.top_task_demos = top_task_demos if int(top_task_demos or 0) > 0 else top_tasks
        self.top_milestones = top_milestones
        self.max_history_steps = max_history_steps
        self.max_segment_steps = max_segment_steps
        # Kept for backward compatibility / ablations; progress checking uses LLM by default.
        self.milestone_check_mode = str(milestone_check_mode or "llm").strip().lower()
        # Debug/trace: last milestone progress check.
        self.last_progress_check: Dict[str, Any] = {}
        # Debug/trace: last retrieved similar tasks (for optional task-level demonstrations).
        self.last_task_examples: List[Dict[str, Any]] = []

    def build_milestone_guide(self, task_text: str, task_type: Optional[str] = None) -> List[str]:
        if not self.library.has_data():
            return []
        # 用新任务和库中的任务做对比，找到相似的任务
        examples = self.library.retrieve_similar_tasks(
            task_text,
            top_k=self.top_tasks,
            task_type=task_type,
        )
        self.last_task_examples = list(examples or [])
        example_blocks = []
        for example in examples:
            guide = example.get("milestone_guide", [])
            if not guide:
                continue
            example_blocks.append(
                "Task: {task}\nMilestone action guide:\n{guide}".format(
                    task=example.get("task"),
                    guide=format_milestone_guide(guide),
                )
            )
        # 把每个找出来的案例拼成这种格式文本：
        # Task: Put a hot apple in fridge
        # Milestone action guide:
        # 1. Find apple
        # 2. Heat apple
        # 3. ...

        prompt_template = _load_prompt(prompt_dir=self.prompt_dir, filename="progress_memory_milestone_guide.txt")
        prompt = prompt_template.format(EXAMPLES="\n\n".join(example_blocks), TASK=task_text)

        response = get_chat(
            prompt=prompt,
            model=self.model_name,
            temperature=0.0,
            max_tokens=256,
        )
        parsed = parse_milestone_guide(response)
        if parsed:
            guide = sanitize_milestone_guide(task_text, parsed)
            return guide[:10] if len(guide) > 10 else guide
        if examples:
            guide = examples[0].get("milestone_guide", [])
            if guide:
                guide2 = sanitize_milestone_guide(task_text, guide)
                return guide2[:10] if len(guide2) > 10 else guide2
            return []
        return []

    def _determine_current_milestone_heuristic(
        self,
        task_text: str,
        trajectory: List[Dict[str, str]],
        milestone_guide: List[str],
        current_milestone_idx: int,
    ) -> int:
        """Decide whether to advance to the next milestone using explicit, evidence-based state.

        This intentionally avoids LLM self-evaluation. A milestone is considered complete only when
        the simulator OBSERVATIONS provide explicit evidence (e.g., "You clean the pan ...").
        """
        if not milestone_guide or not trajectory:
            self.last_progress_check = {"decision": "NO", "evidence_step": 0, "raw": ""}
            return current_milestone_idx

        # Clamp to valid range.
        current_milestone_idx = max(0, min(current_milestone_idx, len(milestone_guide) - 1))
        if current_milestone_idx >= len(milestone_guide) - 1:
            # Already at last milestone; no further progress check is meaningful.
            self.last_progress_check = {"decision": "NO", "evidence_step": 0, "raw": ""}
            return current_milestone_idx

        # 解析当前milestone 的具体内容
        current_milestone = str(milestone_guide[current_milestone_idx] or "").strip()
        sig = parse_milestone_signature(current_milestone)
        kind = sig.get("kind")
        obj = _normalize_token(sig.get("object") or "")
        tgt = _normalize_token(sig.get("target") or "")
        ordinal = _extract_milestone_ordinal(current_milestone)

        # Derive explicit state from full trajectory (cheap; <= 49 steps).
        state = TrajectoryState.from_trajectory(trajectory)

        recent_steps = trajectory[-self.max_history_steps:]

        def _picked_item_norm(token: str, k: int) -> Optional[str]:
            if not token:
                return None
            items = state.picked_order_by_token.get(token) or []
            if len(items) >= k:
                return items[k - 1]
            return None

        advance = False
        evidence_step = 0

        if kind == _MILESTONE_KIND_FIND_PICK and obj:
            # Completed if we have picked up >= ordinal distinct instances of this object.
            items = state.picked_order_by_token.get(obj) or []
            if len(items) >= ordinal:
                advance = True
                evidence_step = _evidence_step_index(
                    recent_steps,
                    lambda _a, o: (not _obs_is_failure(o))
                    and re.search(rf"\byou\s+pick\s+up\s+the\s+{re.escape(obj)}\b", o.lower())
                    is not None,
                )

        elif kind in {_MILESTONE_KIND_CLEAN, _MILESTONE_KIND_HEAT, _MILESTONE_KIND_COOL} and obj:
            required_item = _picked_item_norm(obj, ordinal) or (state.picked_order_by_token.get(obj) or [None])[-1]
            prop = {
                _MILESTONE_KIND_CLEAN: "cleaned",
                _MILESTONE_KIND_HEAT: "heated",
                _MILESTONE_KIND_COOL: "cooled",
            }[kind]
            if required_item and prop in state.item_properties.get(required_item, set()):
                advance = True
                verb = {"cleaned": "clean", "heated": "heat", "cooled": "cool"}[prop]
                evidence_step = _evidence_step_index(
                    recent_steps,
                    lambda _a, o: (not _obs_is_failure(o))
                    and re.search(rf"\byou\s+{verb}\s+the\s+{re.escape(obj)}\b", o.lower())
                    is not None,
                )

        elif kind == _MILESTONE_KIND_PUT and obj:
            required_item = _picked_item_norm(obj, ordinal) or (state.picked_order_by_token.get(obj) or [None])[-1]
            if required_item:
                for ev in reversed(state.put_events):
                    if ev.get("item_norm") != required_item:
                        continue
                    if not _matches_put_target(tgt, ev):
                        continue
                    advance = True
                    break
            if advance:
                evidence_step = _evidence_step_index(
                    recent_steps,
                    lambda _a, o: (not _obs_is_failure(o))
                    and re.search(rf"\byou\s+(?:put|move)\s+the\s+{re.escape(obj)}\b", o.lower())
                    is not None,
                )

        elif kind == _MILESTONE_KIND_OPEN and tgt:
            if any(_matches_location_target(tgt, loc_norm) and state.location_status.get(loc_norm) == "open" for loc_norm in state.opened_locations):
                advance = True
                evidence_step = _evidence_step_index(
                    recent_steps,
                    lambda _a, o: (not _obs_is_failure(o))
                    and (" is open" in o.lower())
                    and (tgt in _normalize_token(o.lower())),
                )

        elif kind == _MILESTONE_KIND_GOTO and tgt:
            # Two cases:
            # - target is a location type (e.g., fridge/sinkbasin): reaching any instance satisfies.
            # - target is an item type (e.g., desklamp): reaching a location where that item is visible satisfies.
            if tgt in _ALFWORLD_LOCATION_TYPES:
                if _matches_location_target(tgt, state.current_location_norm):
                    advance = True
                    evidence_step = _evidence_step_index(
                        recent_steps,
                        lambda a, o: a.lower().startswith("go to")
                        and (not _obs_is_failure(o))
                        and _matches_location_target(tgt, _entity_phrase_to_token_and_norm(_extract_location_from_observation(o) or "")[1] if _extract_location_from_observation(o) else None),
                    )
            else:
                loc_norm = state.current_location_norm
                if loc_norm and tgt in state.visible_items_by_location.get(loc_norm, set()):
                    advance = True
                    evidence_step = _evidence_step_index(
                        recent_steps,
                        lambda _a, o: (not _obs_is_failure(o)) and (tgt in [t for t, _n in _extract_visible_items_from_observation(o)]),
                    )

        elif kind == _MILESTONE_KIND_USE and obj:
            required_item = _picked_item_norm(obj, ordinal) or (state.picked_order_by_token.get(obj) or [None])[-1]
            # Lamps are typically not picked up; allow "use desklamp 1" evidence without prior take.
            if required_item and "used" in state.item_properties.get(required_item, set()):
                advance = True
            else:
                # Fallback: any observed "turn on the <obj>" counts.
                evidence_step = _evidence_step_index(
                    recent_steps,
                    lambda _a, o: (not _obs_is_failure(o))
                    and re.search(rf"\byou\s+turn\s+on\s+the\s+{re.escape(obj)}\b", o.lower())
                    is not None,
                )
                advance = evidence_step > 0

        if advance and evidence_step <= 0:
            # If we advanced based on full-trajectory state but couldn't find a matching recent evidence step,
            # set evidence_step to 1 to keep logs readable.
            evidence_step = 1

        if not advance:
            self.last_progress_check = {"decision": "NO", "evidence_step": 0, "raw": "NO 0"}
            return current_milestone_idx

        self.last_progress_check = {
            "decision": "YES",
            "evidence_step": int(evidence_step),
            "raw": f"YES {int(evidence_step)}",
        }
        return current_milestone_idx + 1

    def _determine_current_milestone_llm(
        self,
        task_text: str,
        trajectory: List[Dict[str, str]],
        milestone_guide: List[str],
        current_milestone_idx: int,
    ) -> int:
        """Simplified LLM-based milestone progress checker that trusts LLM judgment."""
        if not milestone_guide or not trajectory:
            self.last_progress_check = {"decision": "NO", "evidence_step": 0, "raw": "NO 0"}
            return current_milestone_idx

        
        current_milestone_idx = max(0, min(current_milestone_idx, len(milestone_guide) - 1))
        if current_milestone_idx >= len(milestone_guide) - 1:
            self.last_progress_check = {"decision": "NO", "evidence_step": 0, "raw": "NO 0"}
            return current_milestone_idx

        recent_steps = trajectory[-self.max_history_steps :] if self.max_history_steps else list(trajectory)

        def _format_recent_steps(steps: List[Dict[str, str]]) -> str:
            lines: List[str] = []
            for i, step in enumerate(steps, 1):
                action = str(step.get("action", "") or "").strip()
                obs = str(step.get("observation", "") or "").strip()
                lines.append(f"{i}. action: {action}\n   observation: {obs}")
            return "\n".join(lines).strip()

        prompt_template = _load_prompt(
            prompt_dir=self.prompt_dir,
            filename="progress_memory_milestone_progress.txt",
        )

        prompt = prompt_template.format(
            TASK=task_text,
            GUIDE=format_milestone_guide(milestone_guide),
            CUR_NUM=current_milestone_idx + 1,
            NUM=len(milestone_guide),
            CUR_MILESTONE=str(milestone_guide[current_milestone_idx] or "").strip(),
            TRAJECTORY=_format_recent_steps(recent_steps),
        )

        response = get_chat(
            prompt=prompt,
            model=self.model_name,
            temperature=0.0,
            max_tokens=512,
        )

        # Parse LLM response
        try:
            parsed = _safe_json_loads(str(response or ""))
        except Exception:
            self.last_progress_check = {
                "decision": "NO",
                "evidence_step": 0,
                "raw": "NO 0",
                "llm_raw": str(response or "")[:500],
            }
            return current_milestone_idx

        if not isinstance(parsed, dict):
            self.last_progress_check = {
                "decision": "NO",
                "evidence_step": 0,
                "raw": "NO 0",
                "llm_raw": str(response or "")[:500],
            }
            return current_milestone_idx

        # Extract next_milestone_idx
        next_idx = parsed.get("next_milestone_idx", current_milestone_idx)
        if isinstance(next_idx, str):
            next_idx = int(next_idx.strip()) if next_idx.strip().isdigit() else current_milestone_idx
        next_idx = max(current_milestone_idx, min(int(next_idx), len(milestone_guide) - 1))

        # Extract evidence_step
        evidence_step = parsed.get("evidence_step", 0)
        if isinstance(evidence_step, str):
            evidence_step = int(evidence_step.strip()) if evidence_step.strip().isdigit() else 0

        # Build progress check info
        decision = "YES" if next_idx > current_milestone_idx else "NO"
        raw = f"{decision} {evidence_step if decision == 'YES' else 0}"
        self.last_progress_check = {
            "decision": decision,
            "evidence_step": evidence_step if decision == "YES" else 0,
            "raw": raw,
            "reason": str(parsed.get("reason", "") or "").strip(),
            "evidence": str(parsed.get("evidence", "") or "").strip(),
            "llm_raw": str(response or "")[:500],
        }
        return next_idx

    def determine_current_milestone(
        self,
        task_text: str,
        trajectory: List[Dict[str, str]],
        milestone_guide: List[str],
        current_milestone_idx: int,
    ) -> int:
        """Decide whether to advance to the next milestone using an LLM-based judge.

        To reduce brittleness from free-form milestone text (LLM-generated), we rely on an LLM
        to judge progress. The prompt still enforces: only advance with explicit evidence from
        OBSERVATIONS, and only move forward.
        """
        return self._determine_current_milestone_llm(
            task_text=task_text,
            trajectory=trajectory,
            milestone_guide=milestone_guide,
            current_milestone_idx=current_milestone_idx,
        )

    def summarize_state(
        self,
        trajectory: List[Dict[str, str]],
        milestone_guide: List[str],
        milestone_idx: int,
        max_list_items: int = 6,
    ) -> str:
        """Return a compact, explicit state summary for grounding the action model."""
        if not trajectory:
            return ""
        state = TrajectoryState.from_trajectory(trajectory)

        at = state.current_location or "unknown"
        holding = state.held_item or "empty"
        holding_token = state.held_item_token or ""

        opened = sorted(state.opened_locations)
        opened_phrases = [state.location_phrase_by_norm.get(n, n) for n in opened][:max_list_items]

        # Current milestone focus (object/target + any evidence we can provide).
        milestone_idx = max(0, min(milestone_idx, max(0, len(milestone_guide) - 1)))
        milestone = str(milestone_guide[milestone_idx] or "").strip()
        sig = parse_milestone_signature(milestone)
        kind = sig.get("kind")
        obj = _normalize_token(sig.get("object") or "")
        tgt = _normalize_token(sig.get("target") or "")
        ordinal = _extract_milestone_ordinal(milestone)

        focus_lines: List[str] = []

        if kind == _MILESTONE_KIND_FIND_PICK and obj:
            have_n = len(state.picked_order_by_token.get(obj, []))
            focus_lines.append(f"milestone=find_pick({obj}) need={ordinal} have={have_n}")
        elif kind in {_MILESTONE_KIND_CLEAN, _MILESTONE_KIND_HEAT, _MILESTONE_KIND_COOL} and obj:
            required = (state.picked_order_by_token.get(obj) or [None])[-1]
            if required and len(state.picked_order_by_token.get(obj, [])) >= ordinal:
                required = state.picked_order_by_token[obj][ordinal - 1]
            prop = {
                _MILESTONE_KIND_CLEAN: "cleaned",
                _MILESTONE_KIND_HEAT: "heated",
                _MILESTONE_KIND_COOL: "cooled",
            }[kind]
            has = bool(required and prop in state.item_properties.get(required, set()))
            focus_lines.append(f"milestone={kind}({obj}) item={required or 'unknown'} evidence={prop}={has}")
        elif kind == _MILESTONE_KIND_PUT and obj:
            required = (state.picked_order_by_token.get(obj) or [None])[-1]
            if required and len(state.picked_order_by_token.get(obj, [])) >= ordinal:
                required = state.picked_order_by_token[obj][ordinal - 1]
            # Known placements for that item.
            placed = False
            placed_to = ""
            for ev in reversed(state.put_events):
                if required and ev.get("item_norm") == required:
                    placed = True
                    placed_to = state.location_phrase_by_norm.get(ev.get("target_norm", ""), ev.get("target_norm", ""))
                    break
            focus_lines.append(
                f"milestone=put({obj}->{tgt or 'any'}) item={required or 'unknown'} placed={placed}{(' to '+placed_to) if placed_to else ''}"
            )
        elif kind == _MILESTONE_KIND_GOTO and tgt:
            if tgt in _ALFWORLD_LOCATION_TYPES:
                focus_lines.append(f"milestone=goto({tgt}) at={state.current_location_norm or 'unknown'}")
            else:
                seen_at = []
                for loc_norm, items in state.visible_items_by_location.items():
                    if tgt in items:
                        seen_at.append(state.location_phrase_by_norm.get(loc_norm, loc_norm))
                seen_at = seen_at[:max_list_items]
                visible_here = bool(state.current_location_norm and tgt in state.visible_items_by_location.get(state.current_location_norm, set()))
                focus_lines.append(
                    f"milestone=goto_item({tgt}) visible_here={visible_here} seen_at={', '.join(seen_at) if seen_at else 'unknown'}"
                )
        elif kind == _MILESTONE_KIND_OPEN and tgt:
            focus_lines.append(f"milestone=open({tgt}) opened={len(state.opened_locations)}")
        elif kind == _MILESTONE_KIND_USE and obj:
            focus_lines.append(f"milestone=use({obj}) holding={holding_token or 'none'}")

        focus = ("; ".join(focus_lines)).strip()
        if focus:
            focus = "Focus: " + focus

        opened_part = ", ".join(opened_phrases) if opened_phrases else "none"
        return (
            "State (explicit; trust OBSERVATIONS only):\n"
            f"- At: {at}\n"
            f"- Holding: {holding}{(' ('+holding_token+')') if holding_token else ''}\n"
            f"- Opened: {opened_part}\n"
            f"- Visited count: {len(state.visited_locations)}\n"
            + (f"- {focus}\n" if focus else "")
        ).strip()

    def build_step_hint(
        self,
        task_text: str,
        trajectory: List[Dict[str, str]],
        milestone_guide: List[str],
        milestone_idx: int,
        exclude_task_id: Optional[str] = None,
    ) -> Tuple[str, Optional[int]]:
        if not milestone_guide:
            return "", None

        # 目前处在的milestone
        milestone_idx = max(0, min(milestone_idx, len(milestone_guide) - 1))
        current_milestone = milestone_guide[milestone_idx]

        available_locations = extract_available_locations_from_trajectory(trajectory or [])
        sig = parse_milestone_signature(current_milestone)
        require_target = bool(sig.get("target")) and sig.get("kind") in {
            _MILESTONE_KIND_PUT,
            _MILESTONE_KIND_GOTO,
            _MILESTONE_KIND_OPEN,
        }

        # 用当前的milestone和库中的milestone做对比，找到相似的milestone
        similar = self.library.retrieve_similar_milestones(
            current_milestone,
            top_k=self.top_milestones,
            exclude_task_ids=[exclude_task_id] if exclude_task_id else None,
            task_type=None,
            infer_type_if_missing=False,
            distinct_task=True,
            require_kind_match=True,
            require_object_match=False,
            require_target_match=False,
            use_evidence_bonus=False,
        )
        segments = [item.get("segment", []) for item in similar if item.get("segment")]

        prompt_template = _load_prompt(prompt_dir=self.prompt_dir, filename="progress_memory_step_hint.txt")

        prompt = prompt_template.format(
            TASK=task_text,
            TRAJECTORY=format_trajectory_steps(trajectory, max_steps=self.max_history_steps),
            GUIDE=format_milestone_guide(milestone_guide),
            AVAILABLE_LOCATIONS=format_available_locations(available_locations),
            SIMILAR=format_segments(segments, max_segment_steps=self.max_segment_steps, mask_ids=True),
        )

        hint = get_chat(
            prompt=prompt,
            model=self.model_name,
            temperature=0.0,
            max_tokens=256,
        ).strip()

        hinted_idx = parse_hint_milestone_index(hint, milestone_guide)
        return hint, hinted_idx

    def build_local_fewshot(
        self,
        task_text: str,
        trajectory: List[Dict[str, str]],
        milestone_guide: List[str],
        milestone_idx: int,
        exclude_task_id: Optional[str] = None,
        max_demo_steps: int = 10,
    ) -> str:
        """Retrieve similar milestone segments as lightweight few-shot examples.

        This avoids an extra LLM call for hint generation; segments are de-IDed to reduce copying
        demo-specific targets (e.g., 'shelf 2') into the current environment.

        Note: if no matching milestone segments exist (after kind/target filters),
        this returns an empty string (no fallback binding).

        max_demo_steps:
        - > 0: cap each retrieved segment to at most this many steps (also bounded by self.max_segment_steps)
        - <= 0: include the full stored segment without truncation
        """
        if not milestone_guide:
            return ""
        milestone_idx = max(0, min(milestone_idx, len(milestone_guide) - 1))
        current_milestone = milestone_guide[milestone_idx]
        sig = parse_milestone_signature(current_milestone)
        require_target = bool(sig.get("target")) and sig.get("kind") in {
            _MILESTONE_KIND_PUT,
            _MILESTONE_KIND_GOTO,
            _MILESTONE_KIND_OPEN,
        }

        similar = self.library.retrieve_similar_milestones(
            current_milestone,
            top_k=self.top_milestones,
            exclude_task_ids=[exclude_task_id] if exclude_task_id else None,
            task_type=None,  # do not restrict milestone retrieval to the same task type
            infer_type_if_missing=False,
            distinct_task=True,
            require_kind_match=True,
            require_object_match=False,
            require_target_match=False,
            use_evidence_bonus=False,
        )
        if not similar:
            return ""

        blocks: List[str] = []
        for idx, ms in enumerate(similar, start=1):
            segment = list(ms.get("segment") or [])
            if not segment:
                continue
            milestone_text = str(ms.get("milestone", "") or "").strip()
            if milestone_text:
                header = f"Demonstration {idx} (Retrieved Milestone: {milestone_text}):"
            else:
                header = f"Demonstration {idx}:"

            if int(max_demo_steps) <= 0:
                limited = segment
            else:
                cap = int(self.max_segment_steps or 0)
                if cap > 0:
                    max_steps = max(1, min(int(max_demo_steps), cap))
                else:
                    max_steps = max(1, int(max_demo_steps))
                limited = segment[:max_steps]

            limited_masked = [
                {
                    "action": _mask_entity_ids(str(step.get("action", ""))),
                    "observation": _mask_entity_ids(str(step.get("observation", ""))),
                }
                for step in limited
            ]
            blocks.append(f"{header}\n{format_trajectory_steps(limited_masked, max_steps=0)}")
        return "\n\n".join(blocks) if blocks else ""


def build_action_prompt(
    env_history: str,
    milestone_guide: List[str],
    hint: str,
    task_text: str = "",
    trajectory: Optional[List[Dict[str, str]]] = None,
    milestone_demos: str = "",
    current_milestone: str = "",
    prompt_dir: Optional[str] = None,
    max_steps: int = 50,
) -> str:
    if prompt_dir:
        prompt_template = _load_prompt(prompt_dir, "progress_memory_action_prompt.txt")
        trajectories_text = format_trajectory_steps(trajectory or [], max_steps=max_steps)
        available_locations = extract_available_locations_from_trajectory(trajectory or [])
        visited_locations = extract_visited_locations_from_trajectory(trajectory or [])
        guide_text = format_milestone_guide(milestone_guide) if milestone_guide else "- None"
        prompt = prompt_template.format(
            MILESTONE_ACTION_GUIDE=guide_text,
            CURRENT_MILESTONE=current_milestone or "- None",
            MILESTONE_LEVEL_DEMONSTRATIONS=milestone_demos or "- None",
            TASK=task_text,
            TRAJECTORIES=trajectories_text,
            STEP_WISE_HINT=hint or "- None",
            AVAILABLE_LOCATIONS=format_available_locations(available_locations),
            VISITED_LOCATIONS=format_visited_locations(visited_locations),
            HISTORY=env_history or "- None",
        )
        return prompt

    prompt = env_history
    if milestone_guide:
        prompt += "\n\nMilestone Action Guide:\n" + format_milestone_guide(milestone_guide)
    if hint:
        prompt += "\n\nStep-Wise Hint:\n" + hint
    prompt += "\n>"
    return prompt
