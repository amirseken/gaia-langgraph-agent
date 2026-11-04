import pytest
from langchain_core.messages import AIMessage

from agent import clean_answer, text_of


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1,234", "1234"),
        ("FunkMonk", "FunkMonk"),
        ("Final Answer: FunkMonk", "FunkMonk"),
        ("The answer is 42", "42"),
        ("answer - Saint Petersburg", "Saint Petersburg"),
        ('"right"', "right"),
        ("`CUB`", "CUB"),
        ("Saint Petersburg.", "Saint Petersburg"),
        ("b,e", "b, e"),
        ("132 ,133,  134", "132, 133, 134"),
        ("Some reasoning first.\nFinal answer: 3", "3"),
        ("  padded  ", "padded"),
        ("", ""),
    ],
)
def test_clean_answer(raw, expected):
    assert clean_answer(raw) == expected


def test_text_of_plain_string():
    assert text_of(AIMessage(content="hello")) == "hello"


def test_text_of_list_of_parts():
    # Gemini can return content as a list of parts instead of a string.
    message = AIMessage(content=[{"type": "text", "text": "Hel"}, {"type": "text", "text": "lo"}])
    assert text_of(message) == "Hello"


def test_text_of_empty():
    assert text_of(AIMessage(content="")) == ""