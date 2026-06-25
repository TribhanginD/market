from agents.synthesis_agent import _inject, _dedupe_list


def test_dedupe_list_trims_and_dedupes():
    items = [" A  b ", "a b", "", None, "c"]
    out = _dedupe_list(items)
    assert out == ["A  b", "c"]


def test_inject_caps_and_dedupes():
    existing = ["one", "two"]
    additions = ["two", "three", "four", "five"]
    out = _inject(existing, additions, cap=2)
    assert out == ["one", "two", "three", "four"]
