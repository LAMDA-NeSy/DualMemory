from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from api_config import get_api_model
from llm_client import make_openai_client
from llm_metrics import record_response
from parsing import parse_craft_action, parse_get_action
from utils import RetryConfig, read_json, read_text, safe_json_loads, with_retries


DEFAULT_EMBEDDING_MODEL = get_api_model("embedding", "all-mpnet-base-v2")
DEFAULT_PROGRESS_MEMORY_PLANNER_MODEL = get_api_model("progress_memory_planner", "gpt-4o-mini")
DEFAULT_PROGRESS_MEMORY_JUDGE_MODEL = get_api_model("progress_memory_judge", DEFAULT_PROGRESS_MEMORY_PLANNER_MODEL)

_SENTENCE_TRANSFORMER_MODELS = {"all-mpnet-base-v2"}
_ST_MODEL_CACHE: Dict[str, Any] = {}

_PROMPT_VAR_RE = re.compile(r"\{([A-Z_][A-Z0-9_]*)\}")
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _render_prompt_template(template: str, variables: Dict[str, Any]) -> str:
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


def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            f"SentenceTransformers is required for embedding model {model_name}. "
            "Install `sentence-transformers` and `torch`."
        ) from exc

    cached = _ST_MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached

    if os.path.isdir(model_name):
        model = SentenceTransformer(model_name)
    else:
        local_path = _sentence_transformer_cache_path(model_name)
        has_weights = any(
            os.path.exists(os.path.join(local_path, fname)) for fname in ("model.safetensors", "pytorch_model.bin")
        )
        if os.path.isdir(local_path) and has_weights:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            model = SentenceTransformer(local_path)
        else:
            model = SentenceTransformer(model_name)

    _ST_MODEL_CACHE[model_name] = model
    return model


def _embed_texts_sentence_transformers(texts: List[str], model: str, batch_size: int = 96) -> List[List[float]]:
    if not texts:
        return []
    model_name = _sentence_transformer_name(model)
    st_model = _load_sentence_transformer(model_name)
    embeddings = st_model.encode(list(texts), batch_size=batch_size, show_progress_bar=False)
    return [list(map(float, emb)) for emb in embeddings]


def _embed_texts_openai(texts: List[str], model: str, batch_size: int = 96) -> List[List[float]]:
    if not texts:
        return []
    client = make_openai_client(model_name=model)
    out: List[List[float]] = []
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


def _embed_texts(texts: List[str], model: str, batch_size: int = 96) -> List[List[float]]:
    if _is_sentence_transformer_model(model):
        return _embed_texts_sentence_transformers(texts, model=model, batch_size=batch_size)
    return _embed_texts_openai(texts, model=model, batch_size=batch_size)


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in vec))
    if norm == 0:
        return [float(v) for v in vec]
    return [float(v) / norm for v in vec]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union else 0.0


def _infer_textcraft_milestone_kind_from_text(text: str) -> Optional[str]:
    low = str(text or "").strip().lower()
    if not low:
        return None
    if re.search(r"\bcraft\b", low):
        return "craft"
    if re.search(r"\b(get|gather|collect|fetch|obtain|find|pick up|pick)\b", low):
        return "get"
    return None


def _infer_textcraft_milestone_kind_from_steps(steps: Iterable[Dict[str, Any]]) -> Optional[str]:
    last_kind: Optional[str] = None
    for step in steps or []:
        action = str((step or {}).get("action") or "").strip()
        if not action:
            continue
        if parse_craft_action(action):
            last_kind = "craft"
            continue
        if parse_get_action(action):
            last_kind = "get"
            continue
    return last_kind


def _infer_textcraft_milestone_kind(
    *,
    milestone_text: str,
    core_steps: Iterable[Dict[str, Any]] | None = None,
    fallback_steps: Iterable[Dict[str, Any]] | None = None,
) -> Optional[str]:
    kind = _infer_textcraft_milestone_kind_from_steps(core_steps or [])
    if kind:
        return kind
    kind = _infer_textcraft_milestone_kind_from_text(milestone_text)
    if kind:
        return kind
    return _infer_textcraft_milestone_kind_from_steps(fallback_steps or [])


class MilestoneLibrary:
    """
    Retrieval library aligned with `webshop/our_design/progress_memory.py`:
    - store normalized `task_embedding` and `milestone_embedding`
    - retrieve by cosine similarity
    - de-duplicate milestone demos by task_id
    """

    def __init__(self, library_path: str, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.library_path = library_path
        self.embedding_model = embedding_model
        self.tasks: List[Dict[str, Any]] = []
        self.milestones: List[Dict[str, Any]] = []
        self._embed_cache: Dict[str, List[float]] = {}
        self._task_tokens: List[set[str]] = []
        self._load()

    def _load(self) -> None:
        p = Path(self.library_path)
        if not p.exists():
            return
        data = read_json(p)
        self.tasks = list(data.get("tasks") or [])
        self.embedding_model = str(data.get("embedding_model") or self.embedding_model).strip() or self.embedding_model

        self.milestones = []
        self._task_tokens = []
        for task in self.tasks:
            task_text = str(task.get("task") or task.get("problem") or "").strip()
            self._task_tokens.append(_tokenize(task_text))
            if "task_embedding" in task:
                task["task_embedding"] = _normalize([float(x) for x in (task.get("task_embedding") or [])])
            trajectory = list(task.get("trajectory") or [])

            for milestone in task.get("milestones", []) or []:
                if not isinstance(milestone, dict):
                    continue
                entry = dict(milestone)
                entry["task_id"] = task.get("task_id")
                entry["task"] = task.get("task") or task.get("problem") or ""
                entry["recipes"] = list(task.get("recipes") or [])
                entry["milestone_tokens"] = _tokenize(str(entry.get("milestone") or ""))
                action_indices = list(entry.get("action_indices") or [])
                core_steps: List[Dict[str, Any]] = []
                for idx in action_indices:
                    try:
                        i = int(idx)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= i < len(trajectory):
                        step = trajectory[i]
                        if isinstance(step, dict):
                            core_steps.append(step)
                entry["core_segment"] = core_steps or list(entry.get("segment") or [])
                entry["milestone_kind"] = (
                    str(entry.get("milestone_kind") or "").strip().lower()
                    or _infer_textcraft_milestone_kind(
                        milestone_text=str(entry.get("milestone") or ""),
                        core_steps=core_steps,
                        fallback_steps=list(entry.get("segment") or []),
                    )
                    or ""
                )
                if "milestone_embedding" in entry:
                    entry["milestone_embedding"] = _normalize([float(x) for x in (entry.get("milestone_embedding") or [])])
                self.milestones.append(entry)

    def has_data(self) -> bool:
        return bool(self.tasks) and bool(self.milestones)

    def embed_query(self, text: str) -> List[float]:
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

    def retrieve_similar_tasks(self, task_text: str, top_k: int = 2) -> List[Dict[str, Any]]:
        if not self.tasks:
            return []
        query = self.embed_query(task_text)
        if query:
            scored = []
            for t in self.tasks:
                emb = t.get("task_embedding")
                if not emb:
                    continue
                scored.append((_cosine_similarity(query, emb), len(t.get("trajectory", []) or []), t))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[: max(0, int(top_k))]
            top.sort(key=lambda x: x[1])  # prefer shorter trajectories among top matches
            if top:
                return [x[2] for x in top]

        q_tokens = _tokenize(task_text)
        scored_fallback = [(_jaccard(q_tokens, toks), len(t.get("trajectory", []) or []), t) for t, toks in zip(self.tasks, self._task_tokens)]
        scored_fallback.sort(key=lambda x: (-x[0], x[1]))
        return [t for _score, _traj_len, t in scored_fallback[: max(0, int(top_k))]]

    def retrieve_similar_milestones(
        self,
        milestone_text: str,
        top_k: int = 2,
        exclude_task_ids: Optional[Iterable[str]] = None,
        require_kind_match: bool = True,
    ) -> List[Dict[str, Any]]:
        if not self.milestones:
            return []
        exclude = set(exclude_task_ids or [])
        query_kind = _infer_textcraft_milestone_kind(milestone_text=str(milestone_text or ""))
        query = self.embed_query(milestone_text)
        candidates = []
        for m in self.milestones:
            if m.get("task_id") in exclude:
                continue
            if require_kind_match and query_kind:
                cand_kind = str(m.get("milestone_kind") or "").strip().lower()
                if cand_kind != query_kind:
                    continue
            candidates.append(m)

        if query:
            scored = []
            for m in candidates:
                emb = m.get("milestone_embedding")
                if not emb:
                    continue
                scored.append((_cosine_similarity(query, emb), m))
            scored.sort(key=lambda x: x[0], reverse=True)

            selected = []
            used_tasks = set()
            for _score, m in scored:
                tid = m.get("task_id")
                if tid in used_tasks:
                    continue
                used_tasks.add(tid)
                selected.append(m)
                if len(selected) >= max(0, int(top_k)):
                    break
            if selected:
                return selected

        q_tokens = _tokenize(milestone_text)
        scored_fallback = [(_jaccard(q_tokens, set(m.get("milestone_tokens") or set())), len(m.get("segment") or []), m) for m in candidates]
        scored_fallback.sort(key=lambda x: (-x[0], x[1]))

        selected = []
        used_tasks = set()
        for _score, _seg_len, m in scored_fallback:
            tid = m.get("task_id")
            if tid in used_tasks:
                continue
            used_tasks.add(tid)
            selected.append(m)
            if len(selected) >= max(0, int(top_k)):
                break
        return selected


def format_milestone_guide(milestones: List[str]) -> str:
    return "\n".join(f"{i + 1}. {m}" for i, m in enumerate(milestones or []) if str(m or "").strip())


def format_trajectory_steps(trajectory: List[Dict[str, str]], max_steps: int = 0) -> str:
    limited = trajectory if max_steps <= 0 else trajectory[-max_steps:]
    lines: List[str] = []
    for i, step in enumerate(limited, start=1):
        action = str(step.get("action", "") or "").strip()
        observation = str(step.get("observation", "") or "").strip()
        lines.append(f"{i}. action: {action}")
        lines.append(f"   observation: {observation}")
    return "\n".join(lines)


def parse_milestone_guide(raw_text: str) -> List[str]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return []
    try:
        parsed = safe_json_loads(raw_text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        if isinstance(parsed, dict) and isinstance(parsed.get("milestones"), list):
            return [str(x).strip() for x in parsed["milestones"] if str(x).strip()]
    except Exception:
        pass

    out: List[str] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\d+\.?\s+(.*)$", line)
        if m:
            line = m.group(1).strip()
        if line:
            out.append(line)
    return out


def _load_prompt(prompt_dir: str, filename: str) -> str:
    path = Path(prompt_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"missing prompt: {path}")
    return read_text(path)


class ProgressMemoryPlanner:
    def __init__(
        self,
        *,
        library_path: str,
        prompt_dir: str,
        planner_model: str = DEFAULT_PROGRESS_MEMORY_PLANNER_MODEL,
        judge_model: str = DEFAULT_PROGRESS_MEMORY_JUDGE_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        top_tasks: int = 2,
        top_milestones: int = 2,
    ) -> None:
        self.library = MilestoneLibrary(library_path, embedding_model=embedding_model)
        self.prompt_dir = str(prompt_dir)
        self.planner_model = str(planner_model or "").strip() or DEFAULT_PROGRESS_MEMORY_PLANNER_MODEL
        self.judge_model = str(judge_model or "").strip() or self.planner_model
        self.top_tasks = int(top_tasks)
        self.top_milestones = int(top_milestones)
        self.last_progress_check: Dict[str, Any] = {}
        self.llm_calls: int = 0

    def has_library(self) -> bool:
        return self.library.has_data()

    def _chat(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        retry: RetryConfig = RetryConfig(),
    ) -> str:
        client = make_openai_client(model_name=model)

        def _once() -> str:
            self.llm_calls += 1
            started_at = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            record_response(
                model=model,
                component="textcraft_progress_memory",
                started_at=started_at,
                response=resp,
            )
            return (resp.choices[0].message.content or "").strip()

        return with_retries(_once, retry=retry, is_retryable=lambda _e: True)

    def build_milestone_guide(self, task_text: str) -> List[str]:
        task_text = str(task_text or "").strip()
        if not task_text:
            return []

        examples = self.library.retrieve_similar_tasks(task_text, top_k=self.top_tasks) if self.library.tasks else []
        example_blocks: List[str] = []
        for ex in examples:
            guide = list(ex.get("milestone_guide") or [])
            if not guide:
                continue
            example_task = str(ex.get("task") or ex.get("problem") or "").strip()
            if not example_task:
                goal = ex.get("goal") or {}
                goal_item = str(goal.get("item") or "").strip()
                goal_count = int(goal.get("count") or 1)
                example_task = f"Goal: craft {goal_count} {goal_item}".strip() if goal_item else "Goal: (unknown)"
            example_blocks.append(
                f"Example Task (Crafting commands + Goal):\n{example_task}\n\nMilestone action guide:\n{format_milestone_guide(guide)}"
            )

        template = _load_prompt(self.prompt_dir, "progress_memory_milestone_guide.txt")
        prompt = _render_prompt_template(
            template,
            {
                "EXAMPLES": "\n\n".join(example_blocks).strip(),
                "TASK": task_text,
            },
        )
        raw = self._chat(prompt=prompt, model=self.planner_model, temperature=0.0, max_tokens=256)
        parsed = parse_milestone_guide(raw)
        if parsed:
            return parsed
        if examples:
            fallback = list(examples[0].get("milestone_guide") or [])
            return [str(x).strip() for x in fallback if str(x).strip()]
        return []

    def retrieve_milestone_hits(
        self,
        milestone_text: str,
        *,
        top_k: Optional[int] = None,
        exclude_task_ids: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.library.has_data():
            return []
        k = self.top_milestones if top_k is None else int(top_k)
        return self.library.retrieve_similar_milestones(
            str(milestone_text or "").strip(),
            top_k=max(0, k),
            exclude_task_ids=exclude_task_ids,
            require_kind_match=True,
        )

    def determine_current_milestone_idx(
        self,
        *,
        task_text: str,
        milestone_guide: List[str],
        current_milestone_idx: int,
        recent_trajectory: List[Dict[str, str]],
        inventory_line: str = "",
    ) -> int:
        if not milestone_guide:
            self.last_progress_check = {}
            return 0

        current_milestone_idx = max(0, min(int(current_milestone_idx), len(milestone_guide) - 1))
        if current_milestone_idx >= len(milestone_guide) - 1:
            self.last_progress_check = {}
            return current_milestone_idx

        template = _load_prompt(self.prompt_dir, "progress_memory_milestone_progress.txt")
        prompt = _render_prompt_template(
            template,
            {
                "TASK": str(task_text or "").strip(),
                "GUIDE": format_milestone_guide(milestone_guide),
                "CUR_NUM": current_milestone_idx + 1,
                "NUM": len(milestone_guide),
                "CUR_MILESTONE": str(milestone_guide[current_milestone_idx] or "").strip(),
                "INVENTORY": str(inventory_line or "").strip(),
                "TRAJECTORY": format_trajectory_steps(recent_trajectory),
            },
        )
        raw = self._chat(prompt=prompt, model=self.judge_model, temperature=0.0, max_tokens=512)
        try:
            parsed = safe_json_loads(raw)
        except Exception:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        self.last_progress_check = dict(parsed)

        next_idx = parsed.get("next_milestone_idx", current_milestone_idx)
        try:
            next_idx = int(next_idx)
        except Exception:
            next_idx = current_milestone_idx

        # Only trust milestone advancement when the judge explicitly marks the
        # current milestone as proven and provides concrete evidence.
        reason = str(parsed.get("reason") or "").strip().lower()
        evidence = str(parsed.get("evidence") or "").strip()
        if next_idx > current_milestone_idx and (reason != "proven" or not evidence):
            next_idx = current_milestone_idx

        # Enforce: stay or advance by exactly one.
        if next_idx < current_milestone_idx:
            next_idx = current_milestone_idx
        if next_idx > current_milestone_idx + 1:
            next_idx = current_milestone_idx + 1
        if next_idx > len(milestone_guide) - 1:
            next_idx = len(milestone_guide) - 1
        return next_idx
