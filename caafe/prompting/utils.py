import re
import json
from typing import Any, Dict, List

# if ```python on response, or ``` on response, or whole response is code, return code
def extract_code(response: str) -> str:
    """Extract code content from text that may contain code blocks.

    Args:
        response: Input text that might contain code blocks
    Returns:
        Extracted code content or original text if no code blocks found
    """
    response = response.strip()
    code_match = re.search(
        r"```(?:\w+)?\s*(.*?)```",
        response,
        re.DOTALL,
    )
    return code_match.group(1).strip() if code_match else response


def _parse_scalar(value: str) -> Any:
    """Parse a string scalar into bool/int/float/None when appropriate."""
    v = value.strip()
    lower = v.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    # int
    if re.fullmatch(r"[-+]?\d+", v):
        try:
            return int(v)
        except ValueError:
            pass
    # float
    if re.fullmatch(r"[-+]?((\d+\.\d*)|(\d*\.\d+)|\d+)([eE][-+]?\d+)?", v):
        try:
            return float(v)
        except ValueError:
            pass
    return v


def parse_tool_calls(response: str) -> List[Dict[str, Any]]:
    """Parse tool-calling blocks from an assistant response.

    Tool calls have the form:
    <tool_name>
      <param1>value</param1>
      <param2>{"json": "value"}</param2>
      ...
    </tool_name>

    Returns a list like:
    [
      {"name": "tool_name", "args": {"param1": value, "param2": parsed_json_or_scalar}}
    ]
    """
    # Find candidate XML-like blocks. We'll then validate each via XML parsing.
    pattern = re.compile(r"<([A-Za-z_][\w\-.]*)>(.*?)</\1>", re.DOTALL)
    tool_calls: List[Dict[str, Any]] = []

    for m in pattern.finditer(response):
        block = m.group(0)
        try:
            # Parse with XML to get direct children as parameters
            import xml.etree.ElementTree as ET

            elem = ET.fromstring(block)
            # Heuristic: consider only elements that have at least one child (i.e., look like a tool call)
            if len(list(elem)) == 0:
                continue

            args: Dict[str, Any] = {}
            for child in elem:
                text = (child.text or "").strip()
                if not text:
                    args[child.tag] = ""
                    continue
                # Try JSON for lists/objects
                if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
                    try:
                        args[child.tag] = json.loads(text)
                        continue
                    except Exception:
                        # fall back to scalar parsing
                        pass
                args[child.tag] = _parse_scalar(text)

            tool_calls.append({"name": elem.tag, "args": args})
        except Exception:
            # If the block isn't valid XML, skip it
            continue

    return tool_calls


def extract_first_tool_call(response: str) -> Dict[str, Any] | None:
    """Convenience helper to return the first parsed tool call or None."""
    calls = parse_tool_calls(response)
    return calls[0] if calls else None