from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


@dataclass
class WebShopSceneGraph:
    """A lightweight WebShop 'scene graph' capturing UI/page state.

    WebShop is a web UI: the most rule-relevant structure is:
    - current page type (init/search/item/item_sub/end)
    - current ASIN (if on item/item_sub/end)
    - clickables visible on the page
    - available result ASINs (search page)
    - available option targets + selected options (item page)

    This intentionally avoids a full DOM graph; it is stable, deterministic, and directly supports
    feasibility rules (e.g., 'click target must be visible', 'Buy Now only on item page', etc.).
    """

    graph: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.graph:
            self.graph = {
                "page": {
                    "page_type": "unknown",
                    "query_string": "",
                    "page_num": 1,
                    "asin": "",
                    "subpage": "",
                    "selected_options": {},
                },
                "ui": {
                    "clickables": [],
                    "asins": [],
                    "option_types": {},
                },
                "history": {
                    "visited_asins": [],
                    "clicked_targets": [],
                    "invalid_actions": [],
                },
            }

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.graph)

    def update_from_env_session(self, session_state: Dict[str, Any]) -> None:
        page = self.graph.setdefault("page", {})
        ui = self.graph.setdefault("ui", {})
        hist = self.graph.setdefault("history", {})

        page_type = session_state.get("page_type") or session_state.get("page_type_inferred") or "unknown"
        page["page_type"] = page_type
        page["query_string"] = session_state.get("query_string", "") or ""
        page["page_num"] = int(session_state.get("page_num", 1) or 1)
        page["asin"] = session_state.get("asin", "") or ""
        page["subpage"] = session_state.get("subpage", "") or ""
        page["selected_options"] = dict(session_state.get("options") or {})

        ui["clickables"] = list(session_state.get("clickables") or [])
        ui["asins"] = list(session_state.get("asins") or [])
        ui["option_types"] = dict(session_state.get("option_types") or {})

        if page.get("asin"):
            visited = list(hist.get("visited_asins") or [])
            visited.append(str(page["asin"]))
            hist["visited_asins"] = _dedupe_keep_order(visited)

    def record_action_outcome(self, *, action_text: str, success: bool) -> None:
        hist = self.graph.setdefault("history", {})
        clicked = list(hist.get("clicked_targets") or [])
        if action_text:
            clicked.append(action_text)
            hist["clicked_targets"] = clicked[-50:]
        if not success:
            invalid = list(hist.get("invalid_actions") or [])
            invalid.append(action_text)
            hist["invalid_actions"] = invalid[-50:]

