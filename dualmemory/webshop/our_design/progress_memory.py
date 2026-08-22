import json
import math
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import openai

from api_config import apply_api_config, get_api_model
from json_utils import fix_and_parse_json
from llm_metrics import record_response
from trajectory_utils import parse_env_history_steps
from utils import get_openai_client


DEFAULT_EMBEDDING_MODEL = get_api_model("embedding", "all-mpnet-base-v2")
DEFAULT_MILESTONE_EXTRACT_MODEL = get_api_model("milestone_extraction", "gpt-4o-mini")

_SENTENCE_TRANSFORMER_MODELS = {"all-mpnet-base-v2"}
_ST_MODEL_CACHE: Dict[str, Any] = {}

_PROMPT_VAR_RE = re.compile(r"\{([A-Z_][A-Z0-9_]*)\}")


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
            "Install sentence-transformers and torch."
        ) from exc

    model = _ST_MODEL_CACHE.get(model_name)
    if model is not None:
        return model

    if os.path.isdir(model_name):
        model = SentenceTransformer(model_name)
    else:
        local_path = _sentence_transformer_cache_path(model_name)
        has_weights = any(
            os.path.exists(os.path.join(local_path, fname))
            for fname in ("model.safetensors", "pytorch_model.bin")
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
    embeddings = st_model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return [list(map(float, emb)) for emb in embeddings]


def _embed_texts(texts: List[str], model: str, batch_size: int = 96) -> List[List[float]]:
    if _is_sentence_transformer_model(model):
        return _embed_texts_sentence_transformers(texts, model=model, batch_size=batch_size)
    apply_api_config()
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = openai.Embedding.create(model=model, input=batch)
        data = sorted(response["data"], key=lambda item: item["index"])
        embeddings.extend([item["embedding"] for item in data])
    return embeddings


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def format_trajectory_steps(trajectory: List[Dict[str, str]], max_steps: int = 0) -> str:
    limited = trajectory if max_steps <= 0 else trajectory[-max_steps:]
    lines = []
    for i, step in enumerate(limited, start=1):
        lines.append(f"{i}. Action: {step.get('action','')}")
        lines.append(f"   Observation: {step.get('observation','')}")
    return "\n".join(lines)


def format_milestone_guide(milestones: List[str]) -> str:
    return "\n".join(f"{i + 1}. {m}" for i, m in enumerate(milestones))


def parse_trial_log(text: str) -> List[Dict[str, Any]]:
    pattern = re.compile(r"#####\s*Environment #(\d+):\s*(.*?)\s*STATUS:\s*(OK|FAIL)\s*#####", re.DOTALL)
    out = []
    for m in pattern.finditer(text or ""):
        out.append({"env_id": int(m.group(1)), "env_history": m.group(2).strip(), "status": m.group(3).strip()})
    return out


def extract_instruction(initial_observation_block: str) -> str:
    lines = [l.strip() for l in (initial_observation_block or "").splitlines()]
    for i, line in enumerate(lines):
        if not line:
            continue
        if line.lower().startswith("instruction"):
            if ":" in line:
                after = line.split(":", 1)[1].strip()
                if after:
                    return after
            for j in range(i + 1, len(lines)):
                nxt = lines[j].strip()
                if not nxt:
                    continue
                if nxt.startswith("[") and nxt.endswith("]"):
                    break
                return nxt
    # Fallback: try regex
    m = re.search(r"instruction\s*:\s*(.+)$", initial_observation_block or "", flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    return (initial_observation_block or "").strip()


def parse_env_history(env_history: str) -> Optional[Dict[str, Any]]:
    # 返回一个字典，包含task、trajectory（一个列表，每一项是一个动作观察对）和actions
    parsed = parse_env_history_steps(env_history)
    if not parsed:
        return None

    initial_block = str(parsed.get("task", "")).strip()
    task_text = extract_instruction(initial_block)

    trajectory = list(parsed.get("trajectory") or [])
    filtered: List[Dict[str, str]] = []
    for step in trajectory:
        action = str(step.get("action", "")).strip()
        obs = str(step.get("observation", "")).strip()
        # 过滤掉think和环境报错
        if action.startswith("think["):
            continue
        # online rules / imagination retries are not real environment steps
        # e.g. "Action in Imagination (attempt 1): search[...]." + "[Rule_...] ..." feedback
        if action.startswith("Action in Imagination"):
            continue
        if obs.startswith("Invalid action!"):
            continue
        if obs.startswith("Invalid action format"):
            continue
        if obs.startswith("[Rule_"):
            continue
        filtered.append({"action": action, "observation": obs})

    return {"task": task_text, "trajectory": filtered, "actions": [s["action"] for s in filtered]}


def _load_prompt(prompt_dir: str, filename: str) -> str:
    path = os.path.join(prompt_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing prompt: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_json_block(text: str) -> str:
    if "[" in text and "]" in text:
        return text[text.index("[") : text.rindex("]") + 1]
    if "{" in text and "}" in text:
        return text[text.index("{") : text.rindex("}") + 1]
    return text


def _safe_json_loads(text: str) -> Any:
    text = _extract_json_block((text or "").strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return json.loads(text)


def extract_milestones(
    *,
    task: str,
    actions: List[str],
    model: str = DEFAULT_MILESTONE_EXTRACT_MODEL,
    prompt_dir: str,
    max_tokens: int = 1024,
) -> List[Dict[str, Any]]:
    if not actions:
        return []
    traj_text = "\n".join(f"{i}. {a}" for i, a in enumerate(actions, start=1))
    prompt_template = _load_prompt(prompt_dir, "progress_memory_milestone_extract.txt")
    prompt = _render_prompt_template(prompt_template, {"TASK": task, "TRAJECTORY": traj_text})

    client = get_openai_client(model)
    started_at = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    record_response(
        model=model,
        component="webshop_progress_memory_extract",
        started_at=started_at,
        response=resp,
    )
    raw = (resp.choices[0].message.content or "").strip()

    try:
        parsed = _safe_json_loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, dict) and "milestones" in parsed:
        parsed = parsed["milestones"]
    if not isinstance(parsed, list):
        return []

    milestones: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        m_text = str(item.get("milestone", "")).strip()
        action_ids = item.get("actions", [])
        if not m_text or not isinstance(action_ids, list):
            continue
        indices: List[int] = []
        for idx in action_ids:
            try:
                indices.append(int(idx))
            except Exception:
                continue
        if not indices:
            continue
        uses_zero = any(i == 0 for i in indices)
        if not uses_zero and min(indices) >= 1:
            indices = [i - 1 for i in indices]
        indices = [i for i in indices if 0 <= i < len(actions)]
        if not indices:
            continue
        milestones.append({"milestone": m_text, "actions": sorted(set(indices))})
    return milestones


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
            data = json.load(f) or {}
        self.tasks = data.get("tasks", [])
        self.embedding_model = data.get("embedding_model", self.embedding_model)

        for task in self.tasks:
            if "task_embedding" in task:
                task["task_embedding"] = _normalize(task["task_embedding"])
            for milestone in task.get("milestones", []):
                entry = dict(milestone)
                entry["task_id"] = task.get("task_id")
                entry["task"] = task.get("task")
                entry["milestone_embedding"] = _normalize(entry.get("milestone_embedding", []))
                self.milestones.append(entry)

    def has_data(self) -> bool:
        return bool(self.tasks) and bool(self.milestones)

    def embed_query(self, text: str) -> List[float]:
        return _normalize(_embed_texts([text], model=self.embedding_model)[0])

    def retrieve_similar_tasks(self, task_text: str, top_k: int = 2) -> List[Dict[str, Any]]:
        if not self.tasks:
            return []
        query = self.embed_query(task_text)
        scored = []
        for t in self.tasks:
            emb = t.get("task_embedding")
            if not emb:
                continue
            scored.append((_cosine_similarity(query, emb), len(t.get("trajectory", [])), t))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        top.sort(key=lambda x: x[1])
        out: List[Dict[str, Any]] = []
        for sim, _, t in top:
            entry = dict(t)
            entry["similarity"] = float(sim)
            out.append(entry)
        return out

    def retrieve_similar_milestones(
        self,
        milestone_text: str,
        top_k: int = 2,
        exclude_task_ids: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.milestones:
            return []
        exclude = set(exclude_task_ids or [])
        query = self.embed_query(milestone_text)
        scored = []
        for m in self.milestones:
            if m.get("task_id") in exclude:
                continue
            emb = m.get("milestone_embedding")
            if not emb:
                continue
            scored.append((_cosine_similarity(query, emb), m))
        scored.sort(key=lambda x: x[0], reverse=True)

        selected = []
        used_tasks = set()
        for sim, m in scored:
            tid = m.get("task_id")
            if tid in used_tasks:
                continue
            used_tasks.add(tid)
            entry = dict(m)
            entry["similarity"] = float(sim)
            selected.append(entry)
            if len(selected) >= top_k:
                break
        return selected


DEFAULT_PROGRESS_MEMORY_MODEL = get_api_model("progress_memory_planner", "gpt-4o-mini")


def _chat(prompt: str, model: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
    client = get_openai_client(model)
    started_at = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    record_response(
        model=model,
        component="webshop_progress_memory",
        started_at=started_at,
        response=resp,
    )
    return (resp.choices[0].message.content or "").strip()


def parse_milestone_guide(raw_text: str) -> List[str]:
    try:
        parsed = _safe_json_loads(raw_text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    # Fallback: parse numbered lines.
    out = []
    for line in (raw_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\\d+\\.?\\s+(.*)$", line)
        if m:
            line = m.group(1).strip()
        if line:
            out.append(line)
    return out


ASIN_RE = re.compile(r"\\bB[0-9A-Z]{9}\\b")
FIXED_CLICK_TARGETS = {
    "Buy Now",
    "Back to Search",
    "Next >",
    "< Prev",
    "Description",
    "Features",
    "Reviews",
    "Attributes",
}


def _mask_action(action: str) -> str:
    action = str(action or "").strip()
    if action.startswith("search["):
        return "search[QUERY]"
    if action.startswith("click[") and action.endswith("]"):
        target = action[len("click[") : -1]
        if target in FIXED_CLICK_TARGETS:
            return action
        if ASIN_RE.fullmatch(target):
            return "click[ASIN]"
        return "click[OPTION]"
    return action


def _mask_text(text: str) -> str:
    return ASIN_RE.sub("[ASIN]", str(text or ""))


def format_segments(
    segments: List[List[Dict[str, str]]],
    max_segment_steps: int = 6,
    mask: bool = True,
) -> str:
    blocks = []
    for idx, segment in enumerate(segments, start=1):
        limited = segment[:max_segment_steps] if max_segment_steps > 0 else segment
        if mask:
            limited = [
                {
                    "action": _mask_action(step.get("action", "")),
                    "observation": _mask_text(step.get("observation", "")),
                }
                for step in limited
            ]
        blocks.append(f"Demonstration {idx}:\n{format_trajectory_steps(limited, max_steps=0)}")
    return "\n\n".join(blocks) if blocks else "- None"


class ProgressMemoryPlanner:
    def __init__(
        self,
        library_path: str,
        prompt_dir: str,
        model_name: str = DEFAULT_PROGRESS_MEMORY_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        top_tasks: int = 2,
        top_milestones: int = 2,
        max_history_steps: int = 8,
        max_segment_steps: int = 6,
    ) -> None:
        self.library = MilestoneLibrary(library_path, embedding_model=embedding_model)
        self.prompt_dir = prompt_dir
        self.model_name = model_name
        self.top_tasks = int(top_tasks)
        self.top_milestones = int(top_milestones)
        self.max_history_steps = int(max_history_steps)
        self.max_segment_steps = int(max_segment_steps)
        self.last_progress_check: Dict[str, Any] = {}
        self.last_guide_examples: List[Dict[str, Any]] = []

    def has_library(self) -> bool:
        return self.library.has_data()

    def build_milestone_guide(self, task_text: str) -> List[str]:
        if not self.library.has_data():
            self.last_guide_examples = []
            return []
        examples = self.library.retrieve_similar_tasks(task_text, top_k=self.top_tasks)
        self.last_guide_examples = [
            {
                "task_id": ex.get("task_id"),
                "task": ex.get("task"),
                "similarity": ex.get("similarity"),
                "milestone_guide": ex.get("milestone_guide"),
            }
            for ex in (examples or [])
        ]
        example_blocks = []
        for ex in examples:
            guide = ex.get("milestone_guide", [])
            if not guide:
                continue
            example_blocks.append(
                "Task: {task}\nMilestone action guide:\n{guide}".format(
                    task=ex.get("task", ""),
                    guide=format_milestone_guide(list(guide)),
                )
            )

        template = _load_prompt(self.prompt_dir, "progress_memory_milestone_guide.txt")
        prompt = _render_prompt_template(template, {"EXAMPLES": "\n\n".join(example_blocks), "TASK": task_text})
        resp = _chat(prompt=prompt, model=self.model_name, temperature=0.0, max_tokens=256)
        parsed = parse_milestone_guide(resp)
        if parsed:
            return parsed
        if examples:
            return list(examples[0].get("milestone_guide", []) or [])
        return []

    def retrieve_milestone_demos(self, milestone_text: str, exclude_task_ids: Optional[Iterable[str]] = None) -> str:
        if not self.library.has_data() or not milestone_text:
            return "- None"
        similar = self.library.retrieve_similar_milestones(
            milestone_text,
            top_k=self.top_milestones,
            exclude_task_ids=exclude_task_ids,
        )
        segments = [m.get("segment") or [] for m in similar if m.get("segment")]
        return format_segments(segments, max_segment_steps=self.max_segment_steps, mask=True)

    def retrieve_milestone_demos_with_meta(
        self,
        milestone_text: str,
        exclude_task_ids: Optional[Iterable[str]] = None,
        *,
        mask: bool = True,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if not self.library.has_data() or not milestone_text:
            return "- None", []
        similar = self.library.retrieve_similar_milestones(
            milestone_text,
            top_k=self.top_milestones,
            exclude_task_ids=exclude_task_ids,
        )
        blocks: List[str] = []
        for idx, m in enumerate(similar, start=1):
            segment = m.get("segment") or []
            limited = segment[: self.max_segment_steps] if self.max_segment_steps > 0 else segment
            if mask:
                limited = [
                    {
                        "action": _mask_action(step.get("action", "")),
                        "observation": _mask_text(step.get("observation", "")),
                    }
                    for step in (limited or [])
                ]
            source_milestone = str(m.get("milestone", "")).strip().replace("\n", " ")
            header = f"Demonstration {idx} (source_milestone: {source_milestone}):"
            body = format_trajectory_steps(limited, max_steps=0).strip()
            if not body:
                body = "- None"
            blocks.append(f"{header}\n{body}")
        demos_text = "\n\n".join(blocks) if blocks else "- None"
        meta: List[Dict[str, Any]] = []
        for rank, m in enumerate(similar, start=1):
            meta.append(
                {
                    "rank": rank,
                    "task_id": m.get("task_id"),
                    "task": m.get("task"),
                    "milestone_id": m.get("milestone_id"),
                    "milestone": m.get("milestone"),
                    "order": m.get("order"),
                    "similarity": m.get("similarity"),
                    "segment_len": len(m.get("segment") or []) if isinstance(m.get("segment"), list) else None,
                }
            )
        return demos_text, meta

    def determine_current_milestone_idx(
        self,
        *,
        task_text: str,
        milestone_guide: List[str],
        current_milestone_idx: int,
        recent_trajectory: List[Dict[str, str]],
    ) -> int:
        if not milestone_guide:
            self.last_progress_check = {}
            return 0
        current_milestone_idx = max(0, min(current_milestone_idx, len(milestone_guide) - 1))

        template = _load_prompt(self.prompt_dir, "progress_memory_milestone_progress.txt")
        prompt = _render_prompt_template(
            template,
            {
                "TASK": task_text,
                "GUIDE": format_milestone_guide(milestone_guide),
                "CUR_NUM": current_milestone_idx + 1,
                "NUM": len(milestone_guide),
                "CUR_MILESTONE": milestone_guide[current_milestone_idx],
                "TRAJECTORY": format_trajectory_steps(recent_trajectory, max_steps=self.max_history_steps),
            },
        )
        raw = _chat(prompt=prompt, model=self.model_name, temperature=0.0, max_tokens=256)
        try:
            parsed = _safe_json_loads(raw)
        except Exception:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        self.last_progress_check = parsed
        next_idx = parsed.get("next_milestone_idx")
        try:
            next_idx = int(next_idx)
        except Exception:
            next_idx = current_milestone_idx

        # Enforce: advance by at most 1.
        if next_idx < current_milestone_idx:
            next_idx = current_milestone_idx
        if next_idx > current_milestone_idx + 1:
            next_idx = current_milestone_idx + 1
        if next_idx > len(milestone_guide) - 1:
            next_idx = len(milestone_guide) - 1
        return next_idx


def build_action_prompt(
    *,
    prompt_dir: str,
    milestone_guide: List[str],
    current_milestone: str,
    milestone_demos: str,
    history_text: str,
    max_history_steps: int = 8,
) -> str:
    template = _load_prompt(prompt_dir, "progress_memory_action_prompt.txt")
    guide_text = format_milestone_guide(milestone_guide) if milestone_guide else "- None"
    prompt = _render_prompt_template(
        template,
        {
            "MILESTONE_ACTION_GUIDE": guide_text,
            "CURRENT_MILESTONE": current_milestone or "- None",
            "MILESTONE_LEVEL_DEMONSTRATIONS": milestone_demos or "- None",
            "HISTORY": history_text or "- None",
        },
    )
    return prompt
