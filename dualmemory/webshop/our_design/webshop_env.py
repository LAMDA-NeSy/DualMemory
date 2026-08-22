import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment


WEBSHOP_URL = "http://127.0.0.1:3000"

ACTION_TO_TEMPLATE = {
    "Description": "description_page.html",
    "Features": "features_page.html",
    "Reviews": "review_page.html",
    "Attributes": "attributes_page.html",
}


def clean_str(text: str) -> str:
    return text.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element) -> bool:
    ignore = {"style", "script", "head", "title", "meta", "[document]"}
    return element.parent.name not in ignore and not isinstance(element, Comment)


def extract_clickables(observation: str) -> list[str]:
    items = []
    for m in re.finditer(r"\[([^\[\]]+)\]", observation or ""):
        items.append(m.group(1).strip())
    seen = set()
    deduped = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        deduped.append(it)
    return deduped


def infer_page_type(observation: str) -> str:
    obs = (observation or "").lower()
    if "your score (min 0.0, max 1.0)" in obs:
        return "end"
    if "page " in obs and "total results" in obs:
        return "search"
    if "price:" in obs and "rating:" in obs:
        return "item"
    if "instruction:" in obs and "search" in obs and "price:" not in obs and "total results" not in obs:
        return "init"
    return "unknown"


def webshop_text(
    *,
    session: str,
    page_type: str,
    query_string: str = "",
    page_num: int = 1,
    asin: str = "",
    options: Optional[Dict[str, str]] = None,
    subpage: str = "",
    base_url: str = WEBSHOP_URL,
    **_: Any,
) -> Tuple[str, Dict[str, Any]]:
    options = options or {}

    if page_type == "init":
        url = f"{base_url}/{session}"
    elif page_type == "search":
        url = f"{base_url}/search_results/{session}/{query_string}/{page_num}"
    elif page_type == "item":
        url = f"{base_url}/item_page/{session}/{asin}/{query_string}/{page_num}/{json.dumps(options)}"
    elif page_type == "item_sub":
        url = f"{base_url}/item_sub_page/{session}/{asin}/{query_string}/{page_num}/{subpage}/{json.dumps(options)}"
    elif page_type == "end":
        url = f"{base_url}/done/{session}/{asin}/{json.dumps(options)}"
    else:
        raise ValueError(f"unknown page_type={page_type!r}")

    html = requests.get(url).text
    html_obj = BeautifulSoup(html, "html.parser")
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))

    observation = ""
    option_type = ""
    options_map: Dict[str, str] = {}
    asins = []
    cnt = 0
    prod_cnt = 0
    just_prod = 0

    for t in visible_texts:
        if t == "\n":
            continue
        if t.replace("\n", "").replace("\\n", "").replace(" ", "") == "":
            continue

        if t.parent.name == "button":
            processed_t = f"\n[{t}] "
        elif t.parent.name == "label":
            if f"'{t}'" in url:
                processed_t = f"[[{t}]]"
            else:
                processed_t = f"[{t}]"
            options_map[str(t)] = option_type
        elif t.parent.get("class") == ["product-link"]:
            processed_t = f"\n[{t}] "
            if prod_cnt >= 3:
                processed_t = ""
            prod_cnt += 1
            asins.append(str(t))
            just_prod = 0
        else:
            processed_t = "\n" + str(t) + " "
            if cnt < 2 and page_type != "init":
                processed_t = ""
            if just_prod <= 2 and prod_cnt >= 4:
                processed_t = ""
            option_type = str(t)
            cnt += 1

        just_prod += 1
        observation += processed_t

    info: Dict[str, Any] = {}
    if options_map:
        info["option_types"] = options_map
    if asins:
        info["asins"] = asins
    if "Your score (min 0.0, max 1.0)" in visible_texts:
        idx = visible_texts.index("Your score (min 0.0, max 1.0)")
        info["reward"] = float(visible_texts[idx + 1])
        observation = "Your score (min 0.0, max 1.0): " + str(visible_texts[idx + 1])

    return clean_str(observation), info


@dataclass
class StepOutput:
    observation: str
    reward: float
    done: bool
    info: Dict[str, Any]
    action_success: bool


class WebShopEnv:
    def __init__(self, base_url: str = WEBSHOP_URL):
        self.base_url = base_url
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def reset(self, session: str) -> StepOutput:
        self.sessions[session] = {"session": session, "page_type": "init", "base_url": self.base_url}
        obs, info = webshop_text(**self.sessions[session])
        self.sessions[session].update(info)
        self.sessions[session]["last_observation"] = obs
        self._refresh_cached_view(session, obs=obs, info=info)
        return StepOutput(observation=obs, reward=0.0, done=False, info=info, action_success=True)

    def step(self, session: str, action: str) -> StepOutput:
        if session not in self.sessions:
            raise ValueError(f"unknown session {session!r}; call reset() first")

        action = (action or "").strip()
        if not action:
            return self._invalid(session, "empty action")

        if action.startswith("think[") and action.endswith("]"):
            obs = "OK."
            # Think does not change the page; keep cached clickables/state.
            return StepOutput(
                observation=obs,
                reward=0.0,
                done=False,
                info={"action_success": True},
                action_success=True,
            )

        if action.startswith("search[") and action.endswith("]"):
            if self.sessions[session].get("page_type") != "init":
                return self._invalid(session, "search only allowed on init page")
            query = action[len("search[") : -1]
            self.sessions[session] = {
                "session": session,
                "page_type": "search",
                "query_string": query,
                "page_num": 1,
                "base_url": self.base_url,
            }
            obs, info = webshop_text(**self.sessions[session])
            self.sessions[session].update(info)
            self.sessions[session]["last_observation"] = obs
            self._refresh_cached_view(session, obs=obs, info=info)
            return StepOutput(
                observation=obs,
                reward=float(info.get("reward", 0.0)),
                done=False,
                info=info | {"action_success": True},
                action_success=True,
            )

        if action.startswith("click[") and action.endswith("]"):
            button = action[len("click[") : -1]
            return self._handle_click(session, button)

        return self._invalid(session, "unrecognized action format")

    def _refresh_cached_view(self, session: str, *, obs: str, info: Dict[str, Any]) -> None:
        state = self.sessions[session]
        state["page_type_inferred"] = infer_page_type(obs)
        state["clickables"] = extract_clickables(obs)
        if "asins" in info:
            state["asins"] = info["asins"]
        if "option_types" in info:
            state["option_types"] = info["option_types"]

    def _invalid(self, session: str, reason: str) -> StepOutput:
        # Keep observation text aligned with the trajectory parser.
        # Store details in info["reason"] for debugging without polluting the text observation.
        obs = "Invalid action!"
        return StepOutput(
            observation=obs,
            reward=0.0,
            done=False,
            info={"action_success": False, "reason": reason},
            action_success=False,
        )

    def _handle_click(self, session: str, button: str) -> StepOutput:
        state = self.sessions[session]
        page_type = state.get("page_type")
        done = False
        observation_override: Optional[str] = None

        clickables = set(state.get("clickables") or [])
        if button not in clickables:
            return self._invalid(session, f"button not visible: {button}")

        if button == "Buy Now":
            if page_type != "item":
                return self._invalid(session, "Buy Now only valid on item page")
            state["page_type"] = "end"
            done = True
        elif button == "Back to Search":
            state.clear()
            state.update({"session": session, "page_type": "init", "base_url": self.base_url})
        elif button == "Next >":
            if page_type != "search":
                return self._invalid(session, "Next > only valid on search page")
            state["page_num"] = int(state.get("page_num", 1)) + 1
        elif button == "< Prev":
            if page_type == "search":
                page_num = int(state.get("page_num", 1))
                if page_num <= 1:
                    return self._invalid(session, "already at first page")
                state["page_num"] = page_num - 1
            elif page_type == "item_sub":
                state["page_type"] = "item"
                state.pop("subpage", None)
            elif page_type == "item":
                state["page_type"] = "search"
                state.pop("asin", None)
                state.pop("options", None)
            else:
                return self._invalid(session, "< Prev invalid here")
        elif button in ACTION_TO_TEMPLATE:
            if page_type != "item":
                return self._invalid(session, f"{button} only valid on item page")
            state["page_type"] = "item_sub"
            state["subpage"] = button
        else:
            if page_type == "search":
                asins = set(state.get("asins") or [])
                if button not in asins:
                    return self._invalid(session, "clicked item not in current results page")
                state["page_type"] = "item"
                state["asin"] = button
            elif page_type == "item":
                option_types: Dict[str, str] = state.get("option_types") or {}
                if button not in option_types:
                    return self._invalid(session, "clicked option not available")
                option_type = option_types[button]
                state.setdefault("options", {})
                state["options"][option_type] = button
                observation_override = f"You have clicked {button}."
            else:
                return self._invalid(session, "click target invalid for this page type")

        obs, info = webshop_text(**state)
        if observation_override:
            obs = observation_override + "\n" + obs
        state.update(info)
        state["last_observation"] = obs
        self._refresh_cached_view(session, obs=obs, info=info)

        reward = float(info.get("reward", 0.0))
        return StepOutput(
            observation=obs,
            reward=reward,
            done=done,
            info=info | {"action_success": True},
            action_success=True,
        )


def build_state_from_env_session(session_state: Dict[str, Any], last_observation: str) -> Dict[str, Any]:
    return {
        "page_type": session_state.get("page_type") or infer_page_type(last_observation),
        "query_string": session_state.get("query_string", ""),
        "page_num": session_state.get("page_num", 1),
        "asin": session_state.get("asin", ""),
        "options": session_state.get("options", {}) or {},
        "subpage": session_state.get("subpage", ""),
        "clickables": list(session_state.get("clickables") or extract_clickables(last_observation)),
        "asins": list(session_state.get("asins") or []),
        "option_types": session_state.get("option_types") or {},
    }


def parse_action(action_text: str) -> Optional[Dict[str, Any]]:
    action_text = (action_text or "").strip()
    if not action_text:
        return None
    m = re.match(r"^(think|search|click)\[(.*)\]$", action_text)
    if not m:
        return None
    name = m.group(1)
    arg = m.group(2)
    if name == "think":
        return {"name": "think", "args": {"text": arg}}
    if name == "search":
        return {"name": "search", "args": {"query": arg}}
    if name == "click":
        return {"name": "click", "args": {"target": arg}}
    return None
