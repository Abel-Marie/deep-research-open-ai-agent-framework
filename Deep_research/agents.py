import os
import json
import inspect
import re
from typing import List, Dict, Any, Callable, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Clients
# -----------------------------

# Primary Client: Google AI Studio (Gemini via OpenAI-compatible API)
gemini_client = OpenAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Fallback Client: OpenRouter
openrouter_client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

FALLBACK_MODELS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "tngtech/deepseek-r1t2-chimera:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

# -----------------------------
# Tool Decorator
# -----------------------------

def function_tool(func: Callable):
    """Decorator to mark a function as a tool for the agent."""
    func._is_tool = True
    sig = inspect.signature(func)

    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    for name, param in sig.parameters.items():
        if name == "context_variables":
            continue

        p_type = "string"
        if param.annotation == int:
            p_type = "integer"
        elif param.annotation == float:
            p_type = "number"
        elif param.annotation == bool:
            p_type = "boolean"

        parameters["properties"][name] = {
            "type": p_type,
            "description": ""
        }

        if param.default == inspect.Parameter.empty:
            parameters["required"].append(name)

    func._tool_schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or f"Function {func.__name__}",
            "parameters": parameters
        }
    }
    return func

# -----------------------------
# Core Data Structures
# -----------------------------

class Agent:
    def __init__(
        self,
        name: str,
        instructions: str,
        tools: Optional[List[Callable]] = None,
        model: str = "gemini-2.5-flash-lite",
        output_type: Optional[Any] = None
    ):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.model = model
        self.output_type = output_type


class RunResult:
    """Normalized return object for Runner.run()"""
    def __init__(self, output=None, message=None):
        self.output = output
        self.message = message

# -----------------------------
# Runner
# -----------------------------

class Runner:
    async def run(
        self,
        agent: Agent,
        messages: List[Dict[str, str]],
        context_variables: Optional[Dict[str, Any]] = None,
        max_iterations: int = 5
    ) -> RunResult:

        if context_variables is None:
            context_variables = {}

        all_messages = [{"role": "system", "content": agent.instructions}] + messages
        tools = [t._tool_schema for t in agent.tools if hasattr(t, "_tool_schema")]

        models_to_try = [agent.model] + FALLBACK_MODELS

        for model_name in models_to_try:
            is_fallback = model_name in FALLBACK_MODELS
            client = openrouter_client if is_fallback else gemini_client

            try:
                iteration = 0

                while iteration < max_iterations:
                    iteration += 1

                    kwargs = {
                        "model": model_name,
                        "messages": all_messages,
                    }

                    if tools:
                        kwargs["tools"] = tools

                    # Only parse when:
                    # - primary model
                    # - structured output expected
                    # - NO tools involved
                    use_parse = (
                        agent.output_type is not None
                        and not is_fallback
                        and not tools
                    )

                    if use_parse:
                        kwargs["response_format"] = agent.output_type

                    try:
                        if use_parse:
                            response = client.beta.chat.completions.parse(**kwargs)
                        else:
                            response = client.chat.completions.create(**kwargs)
                    except Exception as call_error:
                        print(f"DEBUG: API call failed for {model_name}: {call_error}")
                        raise

                    message = response.choices[0].message
                    all_messages.append(message)

                    # -----------------------------
                    # Exit if no tool calls
                    # -----------------------------
                    if not message.tool_calls:

                        if agent.output_type:
                            # Auto-parsed Gemini result
                            if hasattr(message, "parsed") and message.parsed:
                                return RunResult(output=message.parsed)

                            # Manual JSON recovery
                            try:
                                content = message.content or ""

                                json_match = re.search(
                                    r"```(?:json|markdown)?\s*(\{.*?\})\s*```",
                                    content,
                                    re.DOTALL
                                )

                                if json_match:
                                    json_str = json_match.group(1)
                                else:
                                    start = content.find("{")
                                    end = content.rfind("}")
                                    if start != -1 and end != -1:
                                        json_str = content[start:end + 1]
                                    else:
                                        raise ValueError("No JSON object found")

                                parsed = agent.output_type.model_validate_json(json_str)
                                return RunResult(output=parsed)

                            except Exception as parse_error:
                                print(f"DEBUG: JSON parse failed ({model_name}): {parse_error}")
                                return RunResult(message=message)

                        return RunResult(message=message)

                    # -----------------------------
                    # Tool handling
                    # -----------------------------
                    for tool_call in message.tool_calls:
                        if hasattr(tool_call, "function"):
                            t_func = tool_call.function
                            t_id = tool_call.id
                        else:
                            t_func = tool_call["function"]
                            t_id = tool_call["id"]

                        tool_name = t_func.name

                        try:
                            tool_args = json.loads(t_func.arguments)
                        except json.JSONDecodeError as e:
                            print(f"ERROR: Invalid JSON for tool '{tool_name}': {e}")
                            tool_args = {}

                        target_tool = next(
                            (t for t in agent.tools if t.__name__ == tool_name),
                            None
                        )

                        if target_tool:
                            print(f"DEBUG: Agent '{agent.name}' calling tool '{tool_name}'")

                            if "context_variables" in inspect.signature(target_tool).parameters:
                                result = target_tool(**tool_args, context_variables=context_variables)
                            else:
                                result = target_tool(**tool_args)

                            all_messages.append({
                                "role": "tool",
                                "tool_call_id": t_id,
                                "name": tool_name,
                                "content": json.dumps(result, ensure_ascii=False),
                            })

            except Exception as e:
                print(f"ERROR: Model '{model_name}' failed: {e}")
                if model_name == models_to_try[-1]:
                    raise
                continue

        raise RuntimeError(f"All models failed for agent '{agent.name}'")
