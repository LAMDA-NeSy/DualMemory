from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from api_config import get_api_model
from llm_client import LLMConfig, make_chat_llm, make_openai_client
from parsing import (
    infer_action_success,
    is_craft_action,
    is_get_action,
    is_think,
    parse_craft_action,
    parse_get_action,
    parse_goal,
    parse_recipes,
)
from utils import ensure_dir, read_json, read_text, safe_json_loads, write_json


@dataclass(frozen=True)
class Milestone:
    kind: str  # "craft"
    item: str
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "item": self.item, "count": int(self.count)}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Milestone":
        return Milestone(kind=str(d.get("kind") or "craft"), item=str(d.get("item") or ""), count=int(d.get("count") or 1))


_PROMPT_VAR_RE = re.compile(r"\{([A-Z_][A-Z0-9_]*)\}")


def _render_prompt_template(template: str, variables: dict[str, Any]) -> str:
    """Render prompt templates with {VARS} while preserving JSON braces in examples.

    Avoid `str.format()` because templates often contain literal JSON like {"k": "v"}.
    Only placeholders of the form {UPPER_SNAKE_CASE} are substituted.
    """

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            return match.group(0)
        return str(variables[key])

    return _PROMPT_VAR_RE.sub(_replace, template or "")


MILESTONE_EXTRACT_SYSTEM_PROMPT = "You are an expert milestone extractor for TextCraft tasks."
MILESTONE_EXTRACT_PROMPT_TEMPLATE = read_text(Path(__file__).resolve().parent / "prompts" / "planner_system_textcraft.txt")

DEFAULT_EMBEDDING_MODEL = get_api_model("embedding", "text-embedding-3-small")

_SENTENCE_TRANSFORMER_MODELS = {"all-mpnet-base-v2"}
_ST_MODEL_CACHE: dict[str, Any] = {}


def _is_sentence_transformer_model(model: str) -> bool:
    model = (model or "").strip()
    return model in _SENTENCE_TRANSFORMER_MODELS or model.startswith("sentence-transformers/") or os.path.isdir(model)


def _sentence_transformer_name(model: str) -> str:
    if model.startswith("sentence-transformers/"):
        return model.split("/", 1)[1]
    return model


def _sentence_transformer_cache_path(model_name: str) -> str:
    st_cache_root = os.path.expanduser("~/.cache/torch/sentence_transformers")
    return os.path.join(st_cache_root, f"sentence-transformers_{model_name}")


def _load_sentence_transformer(model: str):
    cached = _ST_MODEL_CACHE.get(model)
    if cached is not None:
        return cached
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "SentenceTransformers is required for offline embedding model "
            f"{model!r}. Install `sentence-transformers` and `torch`, or use an OpenAI embedding model."
        ) from exc

    model_name = _sentence_transformer_name(model)
    cache_path = _sentence_transformer_cache_path(model_name)
    st = SentenceTransformer(cache_path if os.path.isdir(cache_path) else model)
    _ST_MODEL_CACHE[model] = st
    return st


def _embed_texts_sentence_transformers(texts: list[str], model: str, batch_size: int = 32) -> list[list[float]]:
    st = _load_sentence_transformer(model)
    vecs = st.encode(list(texts), batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=False)
    return [v.tolist() for v in vecs]


def _embed_texts_openai(texts: list[str], model: str) -> list[list[float]]:
    client = make_openai_client(model_name=model)
    out: list[list[float]] = []
    batch_size = 96
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        resp = client.embeddings.create(model=model, input=batch)
        data = getattr(resp, "data", None) or []
        try:
            data = sorted(data, key=lambda item: int(getattr(item, "index", 0)))
        except Exception:
            pass
        for item in data:
            emb = getattr(item, "embedding", None)
            if emb is None and isinstance(item, dict):
                emb = item.get("embedding")
            if isinstance(emb, list):
                out.append([float(x) for x in emb])
    return out


def _embed_texts(texts: list[str], model: str) -> list[list[float]]:
    if _is_sentence_transformer_model(model):
        return _embed_texts_sentence_transformers(texts, model=model)
    return _embed_texts_openai(texts, model=model)


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= 0:
        return [float(x) for x in vec]
    return [float(x) / norm for x in vec]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _normalize_action_indices(action_ids: list[Any], action_count: int) -> list[int]:
    indices: list[int] = []
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

    return sorted({i for i in indices if 0 <= i < int(action_count)})


def _parse_milestone_extract_llm(raw_text: str, *, action_count: int) -> list[dict[str, Any]]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return []

    parsed: Any = None
    try:
        parsed = safe_json_loads(raw_text)
    except Exception:
        parsed = None

    if isinstance(parsed, dict) and "milestones" in parsed:
        parsed = parsed.get("milestones")

    if not isinstance(parsed, list):
        return []

    out: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        milestone = str(item.get("milestone") or "").strip()
        action_ids = item.get("actions")
        if not milestone or not isinstance(action_ids, list):
            continue
        indices = _normalize_action_indices(action_ids, int(action_count))
        if not indices:
            continue
        out.append({"milestone": milestone, "actions": indices})
    return out


def _infer_last_craft_output(segment: list[dict[str, str]]) -> tuple[str, int]:
    for step in reversed(segment or []):
        parsed = parse_craft_action(str(step.get("action") or ""))
        if not parsed:
            continue
        item = str(parsed.args.get("item") or "").strip()
        count = int(parsed.args.get("count") or 0)
        return item, count
    return "", 0


def _segment_with_context(
    trajectory: list[dict[str, str]],
    *,
    start_idx: int,
    end_idx: int,
    context_steps: int = 1,
) -> list[dict[str, str]]:
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


def _build_milestone_entries(
    *,
    task_id: str,
    extracted: list[dict[str, Any]],
    trajectory: list[dict[str, str]],
) -> list[dict[str, Any]]:
    milestone_entries: list[dict[str, Any]] = []
    cursor = 0

    for m in extracted:
        milestone_text = str(m.get("milestone") or "").strip()
        raw_action_indices = list(m.get("actions") or [])
        if not milestone_text or not raw_action_indices:
            continue

        start_idx = max(int(cursor), int(min(raw_action_indices)))
        end_idx = int(max(raw_action_indices))
        if start_idx > end_idx:
            continue

        action_indices = list(range(start_idx, end_idx + 1))
        segment = _segment_with_context(
            trajectory,
            start_idx=start_idx,
            end_idx=end_idx,
            context_steps=1,
        )
        if not segment:
            continue

        milestone_item, milestone_count = _infer_last_craft_output(segment)

        order = len(milestone_entries)
        milestone_entries.append(
            {
                "milestone_id": f"{task_id}_m{order}",
                "milestone": milestone_text,
                "order": order,
                "action_indices": action_indices,
                "segment": segment,
                "milestone_item": milestone_item,
                "milestone_count": int(milestone_count or 0),
            }
        )
        cursor = end_idx + 1

    return milestone_entries


def _recipe_map(recipes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in recipes or []:
        if not isinstance(r, dict):
            continue
        output = r.get("output") or {}
        item = output.get("item")
        if isinstance(item, str) and item:
            out[item] = r
    return out


def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        return 0
    return int(math.ceil(a / b))


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union else 0.0


def compute_craft_requirements(goal: dict[str, Any], recipes: list[dict[str, Any]]) -> dict[str, int]:
    """
    Return {craftable_item: required_count} for crafting the goal once.

    Assumptions:
    - The allowed recipes are exactly the "Crafting commands" list.
    - Each recipe has a fixed output count; to get more you repeat the command.
    """
    goal_item = str((goal or {}).get("item") or "").strip()
    goal_count = int((goal or {}).get("count") or 1)
    if not goal_item:
        return {}

    by_output = _recipe_map(recipes)
    craftable = set(by_output.keys())

    # Build dependency edges among craftable items: item -> craftable prereqs.
    prereqs: dict[str, set[str]] = {item: set() for item in craftable}
    for out_item, recipe in by_output.items():
        for ing in recipe.get("inputs") or []:
            if not isinstance(ing, dict):
                continue
            ing_item = ing.get("item")
            if isinstance(ing_item, str) and ing_item in craftable:
                prereqs[out_item].add(ing_item)

    # Collect reachable craftable items from goal.
    reachable: set[str] = set()

    def _reach(x: str) -> None:
        if x in reachable:
            return
        if x not in craftable:
            return
        reachable.add(x)
        for p in prereqs.get(x, set()):
            _reach(p)

    _reach(goal_item)
    if goal_item not in reachable:
        return {}

    # Topological order (prereqs first) via DFS postorder.
    visited: set[str] = set()
    topo: list[str] = []

    def _dfs(x: str) -> None:
        if x in visited:
            return
        visited.add(x)
        for p in prereqs.get(x, set()):
            if p in reachable:
                _dfs(p)
        topo.append(x)

    _dfs(goal_item)
    # topo is prereqs-first with goal last.

    # Process in reverse topological order (dependents first) to aggregate demands without repeated rounding.
    demand: dict[str, int] = {goal_item: goal_count}
    produced: dict[str, int] = {}
    for item in reversed(topo):
        need = int(demand.get(item, 0))
        recipe = by_output.get(item)
        if recipe is None or need <= 0:
            continue
        out_count = int((recipe.get("output") or {}).get("count") or 0)
        if out_count <= 0:
            continue
        times = _ceil_div(need, out_count)
        produced[item] = int(times * out_count)
        for ing in recipe.get("inputs") or []:
            if not isinstance(ing, dict):
                continue
            ing_item = ing.get("item")
            ing_count = ing.get("count")
            if isinstance(ing_item, str) and isinstance(ing_count, int) and ing_count > 0:
                demand[ing_item] = int(demand.get(ing_item, 0)) + int(times * ing_count)

    # Return only craftable items' produced counts.
    return {k: int(v) for k, v in produced.items()}


def compute_craft_plan(goal: dict[str, Any], recipes: list[dict[str, Any]]) -> list[Milestone]:
    """
    Produce an ordered list of craft milestones (prereqs first).
    """
    produced = compute_craft_requirements(goal, recipes)
    if not produced:
        return []

    by_output = _recipe_map(recipes)
    craftable = set(by_output.keys())

    visited: set[str] = set()
    order: list[str] = []

    def _dfs(item: str) -> None:
        if item in visited:
            return
        visited.add(item)
        recipe = by_output.get(item)
        if recipe is None:
            return
        for ing in recipe.get("inputs") or []:
            if not isinstance(ing, dict):
                continue
            ing_item = ing.get("item")
            if isinstance(ing_item, str) and ing_item in craftable and ing_item in produced:
                _dfs(ing_item)
        order.append(item)

    goal_item = str(goal.get("item") or "")
    if goal_item in produced:
        _dfs(goal_item)

    milestones: list[Milestone] = []
    for item in order:
        milestones.append(Milestone(kind="craft", item=item, count=int(produced.get(item, 1))))
    return milestones


def format_milestone_guide(guide: list[Milestone]) -> str:
    lines = []
    for i, m in enumerate(guide, 1):
        lines.append(f"{i}. Craft {m.count} {m.item}")
    return "\n".join(lines)


def parse_episode_steps(episode: dict[str, Any]) -> list[dict[str, Any]]:
    steps = episode.get("steps") or []
    if not isinstance(steps, list):
        return []
    out: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "").strip()
        obs = str(step.get("observation") or "").strip()
        if not action or is_think(action):
            continue
        ok = step.get("action_success")
        if ok is None:
            ok = infer_action_success(action, obs)
        out.append({"action": action, "observation": obs, "action_success": bool(ok)})
    return out


def filter_successful_env_actions(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keep only successful get/craft actions (and drop inventory).
    This matches the ALFWorld S2 progress memory library setting: drop failed pairs.
    """
    filtered: list[dict[str, Any]] = []
    for step in steps:
        action = str(step.get("action") or "")
        if not (is_get_action(action) or is_craft_action(action)):
            continue
        if bool(step.get("action_success")) is False:
            continue
        filtered.append({"action": action, "observation": str(step.get("observation") or "")})
    return filtered


def build_milestones_from_plan(
    *,
    guide: list[Milestone],
    trajectory: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """
    Segment a successful trajectory into milestone segments by the last craft action for each milestone item.
    """
    # Find indices of successful craft actions for each milestone item.
    craft_indices: dict[str, list[int]] = {}
    for i, step in enumerate(trajectory):
        action = step.get("action", "")
        parsed = parse_craft_action(action)
        if not parsed:
            continue
        item = str(parsed.args.get("item") or "")
        craft_indices.setdefault(item, []).append(i)

    milestones: list[dict[str, Any]] = []
    cursor = 0
    for order, m in enumerate(guide):
        indices = craft_indices.get(m.item) or []
        if not indices:
            continue
        end = max(indices)
        segment = _segment_with_context(
            trajectory,
            start_idx=cursor,
            end_idx=end,
            context_steps=1,
        )
        action_indices = list(range(cursor, end + 1))
        cursor = end + 1
        milestones.append(
            {
                "milestone": m.to_dict(),
                "milestone_kind": m.kind,
                "milestone_item": m.item,
                "milestone_count": int(m.count),
                "order": int(order),
                "action_indices": action_indices,
                "segment": segment,
            }
        )
    return milestones


class MilestoneLibrary:
    def __init__(self, library_path: str) -> None:
        self.library_path = library_path
        self.tasks: list[dict[str, Any]] = []
        self.embedding_model: str = DEFAULT_EMBEDDING_MODEL
        self._task_tokens: list[set[str]] = []
        self._task_embeddings: list[list[float]] = []
        self._has_embeddings: bool = False
        self._embed_cache: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        if not Path(self.library_path).exists():
            self.tasks = []
            return
        data = read_json(self.library_path)
        self.embedding_model = str(data.get("embedding_model") or DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL
        self.tasks = list(data.get("tasks") or [])
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._task_tokens = []
        self._task_embeddings = []
        self._has_embeddings = False
        self._embed_cache = {}
        for t in self.tasks:
            problem = str(t.get("problem") or "")
            goal = t.get("goal") or {}
            goal_item = str(goal.get("item") or "")
            self._task_tokens.append(_tokenize((goal_item + "\n" + problem).strip()))
            emb = t.get("task_embedding")
            if isinstance(emb, list) and emb:
                try:
                    self._task_embeddings.append([float(x) for x in emb])
                    self._has_embeddings = True
                except Exception:
                    self._task_embeddings.append([])
            else:
                self._task_embeddings.append([])

    def has_data(self) -> bool:
        return bool(self.tasks)

    def _embed_query(self, text: str) -> list[float]:
        key = (text or "").strip()
        if not key:
            return []
        cached = self._embed_cache.get(key)
        if cached is not None:
            return cached
        try:
            vec = _normalize(_embed_texts([key], model=self.embedding_model)[0])
        except Exception:
            vec = []
        self._embed_cache[key] = vec
        return vec

    def retrieve_similar_tasks(self, problem_text: str, *, top_k: int = 2) -> list[dict[str, Any]]:
        if not self.tasks:
            return []
        if self._has_embeddings and any(self._task_embeddings):
            q = self._embed_query(problem_text)
            if q:
                scored: list[tuple[float, int]] = []
                for i, emb in enumerate(self._task_embeddings):
                    if not emb:
                        continue
                    scored.append((_cosine_similarity(q, emb), i))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [self.tasks[i] for _s, i in scored[:top_k]]

        # Fallback: token Jaccard (no embedding available).
        if not self._task_tokens:
            return []
        q_tok = _tokenize(problem_text)
        scored2: list[tuple[float, int]] = [(_jaccard(q_tok, toks), i) for i, toks in enumerate(self._task_tokens)]
        scored2.sort(key=lambda x: x[0], reverse=True)
        return [self.tasks[i] for _s, i in scored2[:top_k]]

    def iter_milestones(self, *, task_filter: Iterable[str] | None = None) -> Iterable[dict[str, Any]]:
        allowed = set(task_filter or [])
        for t in self.tasks:
            task_id = str(t.get("task_id") or "")
            if allowed and task_id not in allowed:
                continue
            for m in t.get("milestones") or []:
                if isinstance(m, dict):
                    yield dict(m, task_id=task_id)

    def retrieve_similar_milestones(self, milestone: Milestone, *, top_k: int = 1) -> list[dict[str, Any]]:
        if not self.tasks:
            return []
        all_milestones = [m for m in self.iter_milestones()]
        if not all_milestones:
            return []

        candidates = [m for m in all_milestones if str(m.get("milestone_item") or "") == milestone.item]
        if not candidates:
            candidates = all_milestones

        any_emb = any(isinstance(m.get("milestone_embedding"), list) and m.get("milestone_embedding") for m in candidates)
        if any_emb:
            q_text = f"Craft {int(milestone.count)} {milestone.item}".strip()
            q = self._embed_query(q_text)
            if q:
                scored: list[tuple[float, int]] = []
                for i, m in enumerate(candidates):
                    emb = m.get("milestone_embedding")
                    if not (isinstance(emb, list) and emb):
                        continue
                    try:
                        emb_f = [float(x) for x in emb]
                    except Exception:
                        continue
                    scored.append((_cosine_similarity(q, emb_f), i))
                scored.sort(key=lambda x: x[0], reverse=True)
                if scored:
                    return [candidates[i] for _s, i in scored[:top_k]]

        candidates.sort(key=lambda m: len(m.get("segment") or []))
        return candidates[:top_k]


def build_progress_memory_library(
    *,
    io_dir: str,
    env_name: str,
    traj_dir: str = "",
    output_path: str,
    only_success: bool = True,
    planner_model: str = "",
    embedding_model: str = "",
) -> dict[str, Any]:
    traj_root = Path(traj_dir) if traj_dir.strip() else (Path(io_dir) / "traj_data" / env_name / "buffer_traj")
    if not traj_root.is_absolute():
        traj_root = Path(io_dir) / traj_root
    tasks: list[dict[str, Any]] = []

    if not traj_root.exists():
        raise FileNotFoundError(f"Missing trajectory root: {traj_root}")

    model_name = planner_model.strip() or get_api_model("milestone_extraction", get_api_model("progress_memory_planner", "gpt-4o-mini"))
    client = make_openai_client(model_name=model_name)
    extractor_llm = make_chat_llm(
        client,
        config=LLMConfig(model=model_name, temperature=0.0, max_tokens=1024),
        system_prompt=MILESTONE_EXTRACT_SYSTEM_PROMPT,
        default_stop=None,
    )

    for child in tqdm(sorted(traj_root.iterdir(), key=lambda p: p.name), desc="Extracting Milestones"):
        if not child.is_dir() or not child.name.startswith("traj_"):
            continue
        ep_path = child / "episode.json"
        if not ep_path.exists():
            continue
        episode = json.loads(ep_path.read_text(encoding="utf-8"))
        # 只保留最终成功的轨迹
        if only_success and not bool(episode.get("success")):
            continue

        # problem = commands (recipes) + goal
        problem = str(episode.get("problem") or "")
        goal = parse_goal(problem) or {"item": "", "count": 1}
        recipes = parse_recipes(problem)

        # 提取完整交互step
        steps = parse_episode_steps(episode)
        # 过滤掉失败的action obs pair
        trajectory = filter_successful_env_actions(steps)
        actions = [str(step.get("action") or "").strip() for step in trajectory if str(step.get("action") or "").strip()]
        if not actions:
            continue
        traj_text = "\n".join(f"{i}. {a}" for i, a in enumerate(actions, start=1))

        prompt = _render_prompt_template(
            MILESTONE_EXTRACT_PROMPT_TEMPLATE,
            {"TASK": problem.strip(), "TRAJECTORY": traj_text},
        ).strip()

        raw = ""
        extracted: list[dict[str, Any]] = []
        for _attempt in range(3):
            raw = extractor_llm(prompt, stop=None)
            cand = _parse_milestone_extract_llm(raw, action_count=len(actions))
            if cand:
                extracted = cand
                break
        if not extracted:
            continue

        milestone_entries = _build_milestone_entries(task_id=child.name, extracted=extracted, trajectory=trajectory)

        if not milestone_entries:
            continue

        tasks.append(
            {
                "task_id": child.name,
                "seed": int(episode.get("seed", -1)),
                # Align with WebShop/ALFWorld library field name.
                "task": problem,
                "problem": problem,
                "goal": goal,
                "recipes": recipes,
                "trajectory": trajectory,
                "milestone_guide": [m.get("milestone") for m in milestone_entries],
                "milestones": milestone_entries,
            }
        )

    emb_model = (embedding_model or "").strip() or DEFAULT_EMBEDDING_MODEL
    if tasks:
        task_texts = [str(t.get("task") or t.get("problem") or "") for t in tasks]
        task_embs = [_normalize(v) for v in _embed_texts(task_texts, model=emb_model)]

        milestone_texts: list[str] = []
        for t in tasks:
            for m in (t.get("milestones") or []):
                milestone_texts.append(str(m.get("milestone") or ""))
        milestone_embs = [_normalize(v) for v in _embed_texts(milestone_texts, model=emb_model)] if milestone_texts else []

        idx = 0
        for t, emb in zip(tasks, task_embs):
            t["task_embedding"] = emb
            for m in (t.get("milestones") or []):
                if idx < len(milestone_embs):
                    m["milestone_embedding"] = milestone_embs[idx]
                idx += 1

    library = {"version": 1, "env_name": env_name, "embedding_model": emb_model, "tasks": tasks}
    ensure_dir(Path(output_path).parent)
    write_json(output_path, library, indent=2)
    return library
