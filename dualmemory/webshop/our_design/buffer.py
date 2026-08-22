import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from api_config import get_api_model
from webshop_env import extract_clickables, infer_page_type, parse_action
from trajectory_utils import parse_env_history_steps
from webshop_sg import WebShopSceneGraph


def _compact_json(obj: Any, max_chars: int = 4000) -> str:
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    if len(raw) <= max_chars:
        return raw
    return raw[: max(0, max_chars - 3)] + "..."


def state_info_transformation(observation: str, scene_graph: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    obs = (observation or "").strip()
    sg = scene_graph or {}
    sg_page_type = sg.get("page", {}).get("page_type")
    sg_clickables = sg.get("ui", {}).get("clickables")
    return {
        "page_type": sg_page_type or infer_page_type(obs),
        "clickables": (
            list(sg_clickables)
            if isinstance(sg_clickables, list) and sg_clickables
            else extract_clickables(obs)
        ),
        "raw_observation": obs,
    }


class Buffer:
    """Parse collected trajectories into (state, action, result) transitions + host rule functions."""

    def __init__(self, io_dir: str, env_name: str = "webshop", model_name: str = "") -> None:
        self.io_dir = io_dir
        self.env_name = env_name
        self.prompt_dir = os.path.join(io_dir, "prompts")
        self.traj_dir = os.path.join(io_dir, "traj_data", env_name)
        self.rules_dir = os.path.join(io_dir, "symbolic_knowledge", env_name)

        self.record_wrong: Dict[str, List[Dict[str, Any]]] = {}
        self.record_correct: Dict[str, List[Dict[str, Any]]] = {}

        self.functions_set: List[Any] = []
        self.rule_code_file = os.path.join(self.rules_dir, "pruned_rules_code.json")
        if os.path.exists(self.rule_code_file):
            self.load_functions_from_file(self.rule_code_file)

    def load_functions_from_file(self, code_file: str) -> None:
        with open(code_file, "r", encoding="utf-8") as f:
            function_strings = json.load(f)

        for func_str in function_strings:
            try:
                exec(func_str, globals())
            except Exception as e:
                print(f"[RuleLoadError] {type(e).__name__}: {e}")
                continue
            m = re.search(r"def\s+(\w+)\s*\(", func_str)
            if not m:
                continue
            func_name = m.group(1)
            if func_name in globals():
                self.functions_set.append(globals()[func_name])

    def run_all_functions(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        scene_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import inspect

        for func in self.functions_set:
            try:
                kwargs = {"state": state or {}, "action": action or {}}
                sig = None
                try:
                    sig = inspect.signature(func)
                except Exception:
                    sig = None
                if sig is not None and "scene_graph" in sig.parameters:
                    kwargs["scene_graph"] = scene_graph or {}
                feedback, success, suggestion = func(**kwargs)
            except Exception as e:
                print(f"[RuleError] {func.__name__}: {type(e).__name__}: {e}")
                continue
            if not bool(success):
                return {
                    "feedback": f"[{func.__name__}] {feedback}".strip(),
                    "success": False,
                    "suggestion": suggestion or "",
                    "rule_id": func.__name__,
                }
        return {"feedback": "OK", "success": True, "suggestion": "", "rule_id": ""}

    def worldcode_get_prediction(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        scene_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.functions_set:
            return {"feedback": "OK", "success": True, "suggestion": "", "rule_id": ""}
        return self.run_all_functions(state, action, scene_graph=scene_graph)

    # 遍历指定目录下的所有轨迹文件，虽然本来也只有一个
    def _iter_buffer_traj_texts(self, traj_dir: str) -> List[Tuple[str, str]]:
        """Return [(file_path, text), ...] for all stored trajectory dumps."""
        results: List[Tuple[str, str]] = []
        if not os.path.isdir(traj_dir):
            return results
        for fname in os.listdir(traj_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(traj_dir, fname)
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                results.append((fpath, f.read()))
        return results

    def _action_result_from_observation(self, observation: str) -> bool:
        obs = (observation or "").strip()
        return not obs.startswith("Invalid action!")

    def string_buffer_for_transitions_pure(self, interval: int, task_id: int, cleanup: bool = True) -> None:
        record_correct_temp: Dict[str, List[Dict[str, Any]]] = {}
        record_wrong_temp: Dict[str, List[Dict[str, Any]]] = {}

        # 遍历这interval个轨迹
        for kk in range(interval):
            traj_idx = task_id + kk
            trajectory_dir = os.path.join(self.traj_dir, "buffer_traj", f"traj_{traj_idx}")
            sg_dir = os.path.join(self.traj_dir, "buffer_SG", f"traj_{traj_idx}")
            # 遍历指定目录下的所有轨迹文件，虽然本来也只有一个
            for fpath, text in self._iter_buffer_traj_texts(trajectory_dir):
                # 返回一个字典，包含task、trajectory（一个列表，每一项是一个动作观察对）和actions
                parsed = parse_env_history_steps(text)
                if not parsed:
                    continue

                sg_path = os.path.join(sg_dir, "sg_" + os.path.basename(fpath))
                sg_history = []
                # 加载场景图
                if os.path.exists(sg_path):
                    try:
                        with open(sg_path, "r", encoding="utf-8") as f:
                            sg_history = json.load(f) or []
                    except Exception:
                        sg_history = []
                transition_counter = 0

                trajectory = list(parsed.get("trajectory") or [])
                state_observation = str(parsed.get("task", "")).strip()
                if trajectory:
                    # state_observation should start as the initial observation block, i.e.,
                    # everything before the first action. In our log format this is `task`.
                    state_observation = str(parsed.get("task", "")).strip()

                # 遍历轨迹的每一步，将其转换为 (State, Action, Result) 形式
                for idx, step in enumerate(trajectory):
                    action_text = str(step.get("action", "")).strip()
                    observation = str(step.get("observation", "")).strip()

                    action = parse_action(action_text)
                    if action is None:
                        continue
                    if action.get("name") == "think":
                        continue
                    
                    sg_info = None
                    if transition_counter < len(sg_history):
                        sg_info = sg_history[transition_counter]

                    # 把文本observation结构化
                    state = state_info_transformation(state_observation, scene_graph=sg_info)
                    # 如果observation是Invalid action!，那么这个action就是错误的
                    action_result = self._action_result_from_observation(observation)
                    if sg_info is None:
                        # Fallback: minimal sg snapshot from current observation only.
                        sg = WebShopSceneGraph()
                        sg.graph["page"]["page_type"] = state.get("page_type", "unknown")
                        sg.graph["ui"]["clickables"] = list(state.get("clickables") or [])
                        sg_info = sg.snapshot()

                    # 构建transition字典
                    transition = {
                        "initial_state": state,
                        "action": action,
                        "action_result": action_result,
                        "sg_info": sg_info,
                        "transition_id": f"{traj_idx}_{idx}",
                        "source_file": os.path.basename(fpath),
                    }
                    transition_counter += 1

                    # 按照action类型存储transition
                    act_key = str(action.get("name") or "unknown")
                    if action_result:
                        self.record_correct.setdefault(act_key, []).append(transition)
                        record_correct_temp.setdefault(act_key, []).append(transition)
                    else:
                        self.record_wrong.setdefault(act_key, []).append(transition)
                        record_wrong_temp.setdefault(act_key, []).append(transition)

                    # Invalid actions do not change the underlying page state.
                    if action_result:
                        state_observation = observation

        os.makedirs(self.traj_dir, exist_ok=True)
        with open(os.path.join(self.traj_dir, "buffer_wrong_all.json"), "w", encoding="utf-8") as f:
            json.dump(self.record_wrong, f, indent=2)
        with open(os.path.join(self.traj_dir, "buffer_correct_all.json"), "w", encoding="utf-8") as f:
            json.dump(self.record_correct, f, indent=2)
        with open(os.path.join(self.traj_dir, "buffer_wrong_temp.json"), "w", encoding="utf-8") as f:
            json.dump(record_wrong_temp, f, indent=2)
        with open(os.path.join(self.traj_dir, "buffer_correct_temp.json"), "w", encoding="utf-8") as f:
            json.dump(record_correct_temp, f, indent=2)

        if cleanup:
            for kk in range(interval):
                traj_idx = task_id + kk
                trajectory_dir = os.path.join(self.traj_dir, "buffer_traj", f"traj_{traj_idx}")
                if os.path.exists(trajectory_dir):
                    shutil.rmtree(trajectory_dir, ignore_errors=True)
                sg_dir = os.path.join(self.traj_dir, "buffer_SG", f"traj_{traj_idx}")
                if os.path.exists(sg_dir):
                    shutil.rmtree(sg_dir, ignore_errors=True)
