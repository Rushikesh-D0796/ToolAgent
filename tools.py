"""Tool implementations and their function-calling schemas for the agent.

Every tool here is either pure Python (calculate, get_current_datetime) or a
plain HTTPS call via `requests` (wikipedia_summary, get_weather) to a free,
key-free public API. No ML libraries, no compiled extensions beyond what
`requests` itself needs — kept deliberately minimal.
"""

import ast
import operator
from datetime import datetime
import requests

# ---------- Calculator ----------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    """Evaluate a parsed arithmetic expression, allowing only numbers and
    basic operators. Deliberately avoids Python's eval() for safety."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression.")


def calculate(expression: str) -> dict:
    try:
        tree = ast.parse(expression, mode="eval")
        return {"result": _safe_eval(tree.body)}
    except Exception as e:
        return {"error": f"Could not evaluate '{expression}': {e}"}


# ---------- Wikipedia ----------

def wikipedia_summary(query: str) -> dict:
    try:
        search_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": query, "limit": 1, "namespace": 0, "format": "json"},
            timeout=10
        )
        search_resp.raise_for_status()
        titles = search_resp.json()[1]
        if not titles:
            return {"error": f"No Wikipedia page found for '{query}'."}

        title = titles[0]
        summary_resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
            timeout=10
        )
        summary_resp.raise_for_status()
        data = summary_resp.json()
        return {"title": data.get("title"), "summary": data.get("extract")}
    except Exception as e:
        return {"error": f"Wikipedia lookup failed: {e}"}


# ---------- Weather (Open-Meteo: free, no API key required) ----------

def get_weather(city: str) -> dict:
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10
        )
        geo_resp.raise_for_status()
        results = geo_resp.json().get("results")
        if not results:
            return {"error": f"Could not find location '{city}'."}

        place = results[0]
        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": place["latitude"], "longitude": place["longitude"], "current_weather": True},
            timeout=10
        )
        weather_resp.raise_for_status()
        current = weather_resp.json().get("current_weather", {})
        return {
            "location": f"{place['name']}, {place.get('country', '')}",
            "temperature_celsius": current.get("temperature"),
            "windspeed_kmh": current.get("windspeed"),
        }
    except Exception as e:
        return {"error": f"Weather lookup failed: {e}"}


# ---------- Date / time ----------

def get_current_datetime() -> dict:
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
    }


# ---------- Schema (sent to Groq) + dispatcher ----------

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression and return the numeric result. Use for any arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '23 * 47 + 12'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_summary",
            "description": "Look up a short factual summary of a topic, person, or place from Wikipedia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic to look up, e.g. 'Albert Einstein'"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a named city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Mumbai'"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date, time, and day of the week.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

_DISPATCH = {
    "calculate": lambda args: calculate(args.get("expression", "")),
    "wikipedia_summary": lambda args: wikipedia_summary(args.get("query", "")),
    "get_weather": lambda args: get_weather(args.get("city", "")),
    "get_current_datetime": lambda args: get_current_datetime(),
}


def execute_tool(name: str, args: dict) -> dict:
    fn = _DISPATCH.get(name)
    if not fn:
        return {"error": f"Unknown tool '{name}'"}
    return fn(args)
