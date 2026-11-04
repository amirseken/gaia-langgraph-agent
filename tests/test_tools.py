import time

import pandas as pd
import pytest

import tools


# ---------------------------------------------------------------- helpers

class FakeResponse:
    """Stands in for a requests.Response."""

    def __init__(self, text="", json_data=None, status_code=200):
        self.text = text
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"{self.status_code} Client Error")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Skip retry/backoff waits so tests run instantly."""
    monkeypatch.setattr(time, "sleep", lambda seconds: None)


# ---------------------------------------------------------------- run_python (real, but local)

def test_run_python_prints_result():
    assert tools.run_python.invoke({"code": "print(sum(range(10)))"}) == "45"


def test_run_python_reports_errors():
    output = tools.run_python.invoke({"code": "print(undefined_name)"})
    assert "NameError" in output


def test_run_python_reminds_to_print():
    output = tools.run_python.invoke({"code": "x = 1 + 1"})
    assert "print()" in output


# ---------------------------------------------------------------- files (real, using a temp folder)

def test_read_text_file(tmp_path):
    path = tmp_path / "script.py"
    path.write_text("print('hi')")
    assert tools.read_text_file.invoke({"file_path": str(path)}) == "print('hi')"


def test_read_table_file_csv(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"item": ["burger", "soda"], "sales": [10.5, 3.0]}).to_csv(path, index=False)
    output = tools.read_table_file.invoke({"file_path": str(path)})
    assert "burger" in output and "10.5" in output


def test_read_table_file_excel_all_sheets(tmp_path):
    path = tmp_path / "data.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="Food", index=False)
        pd.DataFrame({"b": [2]}).to_excel(writer, sheet_name="Drinks", index=False)
    output = tools.read_table_file.invoke({"file_path": str(path)})
    assert "Sheet: Food" in output and "Sheet: Drinks" in output


# ---------------------------------------------------------------- wikipedia_search (mocked)

def test_wikipedia_search_formats_results(monkeypatch):
    fake = {"query": {"search": [{"title": "Mercedes Sosa", "snippet": "Argentine <span>singer</span>"}]}}
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: FakeResponse(json_data=fake))
    output = tools.wikipedia_search.invoke({"query": "Mercedes Sosa"})
    assert "https://en.wikipedia.org/wiki/Mercedes_Sosa" in output
    assert "<span>" not in output  # HTML stripped from snippets


def test_wikipedia_search_no_results(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: FakeResponse(json_data={"query": {"search": []}}))
    assert tools.wikipedia_search.invoke({"query": "nothing"}) == "No Wikipedia results."


def test_wikipedia_search_sends_user_agent(monkeypatch):
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["headers"] = headers
        return FakeResponse(json_data={"query": {"search": []}})

    monkeypatch.setattr(tools.requests, "get", fake_get)
    tools.wikipedia_search.invoke({"query": "x"})
    assert captured["headers"]["User-Agent"] == tools.USER_AGENT


# ---------------------------------------------------------------- visit_webpage (mocked)

def test_visit_webpage_strips_navigation(monkeypatch):
    html = "<html><body><nav>Main menu</nav><main><h1>Title</h1><p>Real content</p></main></body></html>"
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: FakeResponse(text=html))
    output = tools.visit_webpage.invoke({"url": "https://example.com"})
    assert "Real content" in output
    assert "Main menu" not in output


def test_visit_webpage_returns_error_instead_of_raising(monkeypatch):
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: FakeResponse(status_code=403))
    output = tools.visit_webpage.invoke({"url": "https://example.com"})
    assert output.startswith("Error fetching the webpage")


# ---------------------------------------------------------------- web_search (mocked)

class FakeDDGS:
    results = []

    def text(self, query, max_results=8):
        return self.results


def test_web_search_formats_results(monkeypatch):
    import ddgs
    FakeDDGS.results = [{"title": "Result", "href": "https://example.com", "body": "Snippet"}]
    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    output = tools.web_search.invoke({"query": "test"})
    assert "[Result](https://example.com)" in output and "Snippet" in output


def test_web_search_handles_rate_limiting(monkeypatch):
    import ddgs

    class FailingDDGS:
        def text(self, query, max_results=8):
            raise Exception("Ratelimit")

    monkeypatch.setattr(ddgs, "DDGS", FailingDDGS)
    output = tools.web_search.invoke({"query": "test"})
    assert "No results" in output and "wikipedia_search" in output