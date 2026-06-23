from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from .artifacts import load_model_artifact
from .data import ByteTokenizer


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolEvalCase:
    name: str
    prompt: str
    tools: tuple[ToolSpec, ...]
    expected_tool: str | None = None
    expected_args: dict[str, Any] | None = None
    expected_final_substrings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedToolCall:
    raw: str
    valid_json: bool
    name: str | None = None
    arguments: dict[str, Any] | None = None
    error: str | None = None


def tool_eval_cases() -> list[ToolEvalCase]:
    return [
        ToolEvalCase(
            name="weather_single_call",
            prompt="Use the tools if needed.\nUser: What is the weather in London?\nAssistant:",
            tools=(
                ToolSpec(
                    name="get_weather",
                    description="Get current weather for a city.",
                    input_schema={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                ),
            ),
            expected_tool="get_weather",
            expected_args={"location": "London"},
        ),
        ToolEvalCase(
            name="currency_required_args",
            prompt="Use the tools if needed.\nUser: Convert 10 USD to EUR.\nAssistant:",
            tools=(
                ToolSpec(
                    name="convert_currency",
                    description="Convert an amount between currencies.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "amount": {"type": "number"},
                            "base_currency": {"type": "string"},
                            "target_currency": {"type": "string"},
                        },
                        "required": ["amount", "base_currency", "target_currency"],
                    },
                ),
            ),
            expected_tool="convert_currency",
            expected_args={"amount": 10, "base_currency": "USD", "target_currency": "EUR"},
        ),
        ToolEvalCase(
            name="timer_arg_type",
            prompt="Use the tools if needed.\nUser: Set a timer for 5 minutes.\nAssistant:",
            tools=(
                ToolSpec(
                    name="set_timer",
                    description="Set a timer in whole minutes.",
                    input_schema={
                        "type": "object",
                        "properties": {"minutes": {"type": "integer"}},
                        "required": ["minutes"],
                    },
                ),
            ),
            expected_tool="set_timer",
            expected_args={"minutes": 5},
        ),
        ToolEvalCase(
            name="mcp_read_file",
            prompt="Use the MCP tool if needed.\nUser: Read package.json from the repository root.\nAssistant:",
            tools=(
                ToolSpec(
                    name="read_file",
                    description="Read a UTF-8 file from the workspace.",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                ),
            ),
            expected_tool="read_file",
            expected_args={"path": "package.json"},
        ),
        ToolEvalCase(
            name="mcp_search_docs",
            prompt="Use the MCP tool if needed.\nUser: Find docs about CUDA smoke runs.\nAssistant:",
            tools=(
                ToolSpec(
                    name="search_files",
                    description="Search files for a literal query.",
                    input_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                ),
            ),
            expected_tool="search_files",
            expected_args={"query": "CUDA smoke"},
        ),
        ToolEvalCase(
            name="no_tool_creative_request",
            prompt="Use the tools if needed.\nUser: Write a short friendly greeting.\nAssistant:",
            tools=(
                ToolSpec(
                    name="get_weather",
                    description="Get current weather for a city.",
                    input_schema={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                ),
            ),
            expected_tool=None,
            expected_final_substrings=("hello",),
        ),
        ToolEvalCase(
            name="no_hallucinated_news_tool",
            prompt="Use the tools if needed.\nUser: Get the latest news headlines.\nAssistant:",
            tools=(
                ToolSpec(
                    name="calculator",
                    description="Evaluate arithmetic.",
                    input_schema={
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                ),
            ),
            expected_tool=None,
            expected_final_substrings=("not", "available"),
        ),
        ToolEvalCase(
            name="function_response_final_answer",
            prompt=(
                "Tool result received.\n"
                'FUNCTION RESPONSE: {"forecast":"sunny","temperature_f":72}\n'
                "User: What should I know?\nAssistant:"
            ),
            tools=(
                ToolSpec(
                    name="get_weather",
                    description="Get current weather for a city.",
                    input_schema={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                ),
            ),
            expected_tool=None,
            expected_final_substrings=("sunny", "72"),
        ),
    ]


def run_tool_eval_for_artifact(
    artifact_dir: str | Path,
    *,
    output: str | Path | None = None,
    max_new_tokens: int = 160,
    temperature: float = 0.0,
    seed: int = 1337,
) -> dict[str, Any]:
    tokenizer = ByteTokenizer()
    model, manifest = load_model_artifact(artifact_dir, map_location="cpu")

    def generate(prompt: str) -> str:
        torch.manual_seed(seed)
        tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        with torch.no_grad():
            generated = model.generate(tokens, max_new_tokens, temperature)  # type: ignore[attr-defined]
        return tokenizer.decode(generated[0, len(tokens[0]) :].tolist())

    report = run_tool_eval(generate, model_info=manifest.get("model", {}), runtime="artifact")
    report["artifact_dir"] = str(artifact_dir)
    report["generation"] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_tool_eval(
    generate: Callable[[str], str],
    *,
    model_info: dict[str, Any] | None = None,
    runtime: str = "callable",
    cases: list[ToolEvalCase] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    case_results = []
    for case in cases or tool_eval_cases():
        prompt = render_prompt(case)
        completion = generate(prompt)
        parsed = parse_tool_call(completion)
        case_results.append(score_case(case, completion, parsed))
    metrics = aggregate_metrics(case_results)
    return {
        "suite_name": "stage_3_tool_eval_v0",
        "runtime": runtime,
        "model": model_info or {},
        "num_cases": len(case_results),
        "metrics": metrics,
        "cases": case_results,
        "wall_clock_time_sec": time.perf_counter() - start,
    }


def render_prompt(case: ToolEvalCase) -> str:
    tools = [asdict(tool) for tool in case.tools]
    return (
        "### Tool Call Evaluation\n"
        "You may call exactly one tool when the task requires it.\n"
        "If calling a tool, respond with JSON like:\n"
        '<functioncall> {"name":"tool_name","arguments":{...}}\n'
        "If no tool applies or a tool result is already present, answer normally.\n"
        f"### Tools\n{json.dumps(tools, indent=2, sort_keys=True)}\n"
        f"### Conversation\n{case.prompt}\n"
    )


def parse_tool_call(completion: str) -> ParsedToolCall:
    candidate = _extract_json_candidate(completion)
    if candidate is None:
        return ParsedToolCall(raw="", valid_json=False, error="no_json_object")
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        return ParsedToolCall(raw=candidate, valid_json=False, error=str(error))
    normalized = _normalize_tool_payload(payload)
    if normalized is None:
        return ParsedToolCall(raw=candidate, valid_json=True, error="json_not_tool_call")
    name, arguments = normalized
    return ParsedToolCall(raw=candidate, valid_json=True, name=name, arguments=arguments)


def score_case(case: ToolEvalCase, completion: str, parsed: ParsedToolCall) -> dict[str, Any]:
    expected_call = case.expected_tool is not None
    schema = schema_for_tool(case, parsed.name)
    schema_errors = validate_arguments(parsed.arguments or {}, schema) if schema else ["unknown_tool"] if parsed.name else []
    final_matches = [
        item
        for item in case.expected_final_substrings
        if item.lower() in completion.lower()
    ]
    tool_name_correct = bool(expected_call and parsed.name == case.expected_tool)
    argument_match = bool(expected_call and arguments_match(parsed.arguments or {}, case.expected_args or {}))
    schema_passed = bool(expected_call and parsed.valid_json and parsed.name and not schema_errors)
    no_tool_correct = bool(not expected_call and parsed.name is None)
    final_answer_correct = (
        len(final_matches) == len(case.expected_final_substrings)
        if case.expected_final_substrings
        else no_tool_correct
    )
    if expected_call:
        score = sum([tool_name_correct, schema_passed, argument_match]) / 3.0
    else:
        score = sum([no_tool_correct, final_answer_correct]) / 2.0
    return {
        "name": case.name,
        "expected_tool": case.expected_tool,
        "expected_args": case.expected_args or {},
        "completion": completion,
        "parsed_tool_call": asdict(parsed),
        "tool_name_correct": tool_name_correct,
        "argument_schema_passed": schema_passed,
        "argument_match": argument_match,
        "no_tool_correct": no_tool_correct,
        "final_answer_expected": bool(case.expected_final_substrings),
        "final_answer_matches": final_matches,
        "final_answer_correct": final_answer_correct,
        "schema_errors": schema_errors,
        "score": score,
        "passed": score == 1.0,
    }


def aggregate_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    tool_cases = [item for item in case_results if item["expected_tool"] is not None]
    no_tool_cases = [item for item in case_results if item["expected_tool"] is None]
    final_cases = [item for item in no_tool_cases if item["final_answer_expected"]]
    return {
        "overall_score": mean([item["score"] for item in case_results]),
        "num_passed": sum(1 for item in case_results if item["passed"]),
        "tool_call_valid_rate": mean([item["argument_schema_passed"] for item in tool_cases]),
        "tool_name_accuracy": mean([item["tool_name_correct"] for item in tool_cases]),
        "argument_schema_pass_rate": mean([item["argument_schema_passed"] for item in tool_cases]),
        "argument_match_rate": mean([item["argument_match"] for item in tool_cases]),
        "no_tool_precision": mean([item["no_tool_correct"] for item in no_tool_cases]),
        "final_answer_after_tool_rate": mean([item["final_answer_correct"] for item in final_cases]),
    }


def mean(values: list[Any]) -> float:
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def schema_for_tool(case: ToolEvalCase, name: str | None) -> dict[str, Any] | None:
    if name is None:
        return None
    for tool in case.tools:
        if tool.name == name:
            return tool.input_schema
    return None


def validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors = []
    if schema.get("type") != "object":
        return ["unsupported_schema_type"]
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    for key in required:
        if key not in arguments:
            errors.append(f"missing_required:{key}")
    for key, value in arguments.items():
        expected = properties.get(key, {}).get("type")
        if expected and not value_matches_json_type(value, expected):
            errors.append(f"type_mismatch:{key}:{expected}")
    return errors


def value_matches_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return True


def arguments_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        if normalize_scalar(actual[key]) != normalize_scalar(expected_value):
            return False
    return True


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _normalize_tool_payload(payload: Any) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("name"), str):
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return payload["name"], {}
        return payload["name"], arguments if isinstance(arguments, dict) else {}
    for key in ("function", "tool_call"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return _normalize_tool_payload(nested)
    return None


def _extract_json_candidate(text: str) -> str | None:
    function_index = text.lower().find("<functioncall>")
    search_start = function_index if function_index >= 0 else 0
    brace_index = text.find("{", search_start)
    if brace_index < 0:
        brace_index = text.find("{")
    if brace_index < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(brace_index, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_index : index + 1]
    match = re.search(r"\{.*", text[brace_index:], flags=re.DOTALL)
    return match.group(0) if match else None
