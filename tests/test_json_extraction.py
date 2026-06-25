from llm.json_utils import extract_json


def test_extract_json_from_qwen_thinking_prefix():
    raw = """
Thinking Process:

1. Analyze request.
2. Build answer.

{
  "symbol": "MAHABANK",
  "bull_points": ["cheap"],
  "bear_points": ["risk"]
}
"""

    assert extract_json(raw, expected=dict)["symbol"] == "MAHABANK"


def test_extract_json_prefers_later_answer_object():
    raw = """
Thinking Process:
{"scratch": true}

Final:
```json
{"symbol": "HINDZINC", "probability_weighted_return_12m": 0.12}
```
"""

    result = extract_json(raw, expected=dict)
    assert result["symbol"] == "HINDZINC"
    assert result["probability_weighted_return_12m"] == 0.12


def test_extract_json_array_from_wrapped_stage1_response():
    raw = 'Reasoning first. Final JSON:\n[{"symbol":"ABC"},{"symbol":"XYZ"}]'

    result = extract_json(raw, expected=list)
    assert [row["symbol"] for row in result] == ["ABC", "XYZ"]
