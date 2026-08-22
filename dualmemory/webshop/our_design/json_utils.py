import json
import re
from typing import Any, Dict, Union


def _extract_char_position(error_message: str) -> int:
    match = re.search(r"\\(char (\\d+)\\)", error_message)
    if not match:
        raise ValueError("Character position not found in JSONDecodeError message.")
    return int(match.group(1))


def _add_quotes_to_property_names(json_string: str) -> str:
    def replace_func(match):
        return f"\"{match.group(1)}\":"

    property_name_pattern = re.compile(r"(\\w+):")
    corrected = property_name_pattern.sub(replace_func, json_string)
    json.loads(corrected)
    return corrected


def _balance_braces(json_string: str) -> str:
    open_count = json_string.count("{")
    close_count = json_string.count("}")
    while open_count > close_count:
        json_string += "}"
        close_count += 1
    while close_count > open_count:
        json_string = json_string.rstrip("}")
        close_count -= 1
    json.loads(json_string)
    return json_string


def _fix_invalid_escape(json_str: str, error_message: str) -> str:
    while error_message.startswith("Invalid \\escape"):
        bad_loc = _extract_char_position(error_message)
        json_str = json_str[:bad_loc] + json_str[bad_loc + 1 :]
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            error_message = str(e)
    return json_str


def correct_json(json_str: str) -> str:
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        error_message = str(e)

    if error_message.startswith("Invalid \\escape"):
        json_str = _fix_invalid_escape(json_str, error_message)
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            error_message = str(e)

    if error_message.startswith("Expecting property name enclosed in double quotes"):
        json_str = _add_quotes_to_property_names(json_str)
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass

    try:
        return _balance_braces(json_str)
    except Exception:
        return json_str


def fix_and_parse_json(json_str: str) -> Union[str, Dict[Any, Any]]:
    json_str = (json_str or "").replace("\t", "").strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    fixed = correct_json(json_str)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Fallback: take the innermost {...} block.
    try:
        brace_index = json_str.index("{")
        json_str = json_str[brace_index:]
        last_brace_index = json_str.rindex("}")
        json_str = json_str[: last_brace_index + 1]
        return json.loads(json_str)
    except Exception as e:
        raise e

