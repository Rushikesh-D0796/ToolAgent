"""The agent loop: sends the conversation to Groq, executes any tool calls
the model requests, feeds results back, and repeats until a final answer
is produced.
"""

import json
from tools import TOOLS_SCHEMA, execute_tool

MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a helpful AI agent with access to four tools:
- calculate: for any arithmetic or math
- wikipedia_summary: for factual questions about people, places, or topics
- get_weather: for current weather in a named city
- get_current_datetime: for today's date, time, or day of the week

Use a tool whenever it would give a more accurate or up-to-date answer than
your own knowledge. You can use more than one tool if the question needs it.
Once you have what you need, give a clear, concise final answer in plain language.
"""


def run_agent(client, messages, max_steps: int = 5):
    """Runs the tool-calling loop until the model returns a plain-text answer.

    Returns (final_answer, trace) where trace is a list of
    {"tool": name, "args": {...}, "result": {...}} dicts describing every
    tool call that happened along the way.
    """
    trace = []

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.3,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content, trace

        # Record the assistant's tool-call request in the conversation history.
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in message.tool_calls
            ]
        })

        # Execute each requested tool and feed the result back to the model.
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            result = execute_tool(name, args)
            trace.append({"tool": name, "args": args, "result": result})

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": json.dumps(result)
            })

    return "I wasn't able to finish reasoning in the allotted steps — try rephrasing the question.", trace
