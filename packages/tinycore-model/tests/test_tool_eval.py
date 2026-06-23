from __future__ import annotations

from tinycore_model.tool_eval import parse_tool_call, run_tool_eval, tool_eval_cases


def test_parse_tool_call_extracts_functioncall_json() -> None:
    parsed = parse_tool_call('<functioncall> {"name":"get_weather","arguments":{"location":"London"}}')

    assert parsed.valid_json is True
    assert parsed.name == "get_weather"
    assert parsed.arguments == {"location": "London"}


def test_parse_tool_call_decodes_string_arguments() -> None:
    parsed = parse_tool_call(
        '<functioncall> {"name":"get_weather","arguments":"{\\"location\\":\\"London\\"}"}'
    )

    assert parsed.valid_json is True
    assert parsed.name == "get_weather"
    assert parsed.arguments == {"location": "London"}


def test_parse_tool_call_handles_no_tool() -> None:
    parsed = parse_tool_call("Hello, nice to see you.")

    assert parsed.valid_json is False
    assert parsed.name is None
    assert parsed.error == "no_json_object"


def test_run_tool_eval_scores_perfect_scripted_model() -> None:
    answers = {
        "weather_single_call": '<functioncall> {"name":"get_weather","arguments":{"location":"London"}}',
        "currency_required_args": (
            '<functioncall> {"name":"convert_currency","arguments":'
            '{"amount":10,"base_currency":"USD","target_currency":"EUR"}}'
        ),
        "timer_arg_type": '<functioncall> {"name":"set_timer","arguments":{"minutes":5}}',
        "mcp_read_file": '<functioncall> {"name":"read_file","arguments":{"path":"package.json"}}',
        "mcp_search_docs": '<functioncall> {"name":"search_files","arguments":{"query":"CUDA smoke"}}',
        "no_tool_creative_request": "Hello, friend.",
        "no_hallucinated_news_tool": "That news tool is not available.",
        "function_response_final_answer": "It is sunny and 72 degrees.",
    }

    cases = tool_eval_cases()

    def generate(prompt: str) -> str:
        for case in cases:
            if case.prompt in prompt:
                return answers[case.name]
        raise AssertionError("unexpected prompt")

    report = run_tool_eval(generate, cases=cases)

    assert report["metrics"]["overall_score"] == 1.0
    assert report["metrics"]["tool_name_accuracy"] == 1.0
    assert report["metrics"]["argument_schema_pass_rate"] == 1.0
    assert report["metrics"]["no_tool_precision"] == 1.0
    assert report["metrics"]["final_answer_after_tool_rate"] == 1.0


def test_run_tool_eval_penalizes_hallucinated_tool() -> None:
    cases = [case for case in tool_eval_cases() if case.name == "no_hallucinated_news_tool"]

    report = run_tool_eval(
        lambda _prompt: '<functioncall> {"name":"get_news","arguments":{"country":"US"}}',
        cases=cases,
    )

    assert report["metrics"]["overall_score"] == 0.0
    assert report["metrics"]["no_tool_precision"] == 0.0
    assert report["cases"][0]["schema_errors"] == ["unknown_tool"]
