import os
import re

import pandas as pd
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_core.tools import tool

load_dotenv()

USER_AGENT = "GAIA-course-agent/1.0 (https://huggingface.co/eidenspark; educational project)"

# Strongest first; the media tools fall back down this list on quota/overload errors.
GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
]

_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=90_000),
)


def _is_quota_error(e) -> bool:
    s = str(e)
    return "429" in s or "RESOURCE_EXHAUSTED" in s


def _is_temporary_error(e) -> bool:
    s = str(e).lower()
    return any(x in s for x in ["503", "unavailable", "overloaded", "timeout", "timed out"])


def _gemini_generate(contents) -> str:
    """Call Gemini with model fallback; used by the media tools."""
    last_error = None
    for model in GEMINI_MODELS:
        try:
            response = _client.models.generate_content(model=model, contents=contents)
            return response.text or "(the model returned an empty response)"
        except Exception as e:
            last_error = e
            if _is_quota_error(e) or _is_temporary_error(e):
                continue
            raise
    return f"Could not analyze: all models unavailable ({last_error})"


# ---------------------------------------------------------------- search & web

@tool
def web_search(query: str) -> str:
    """Search the web and return the top results with titles, URLs and snippets."""
    import time
    from ddgs import DDGS
    last_error = None
    for attempt in range(3):
        try:
            results = DDGS().text(query, max_results=8)
            if results:
                return "\n\n".join(f"[{r['title']}]({r['href']})\n{r['body']}" for r in results)
        except Exception as e:
            last_error = e
        time.sleep(5 * (attempt + 1))
    return (f"No results (search may be rate-limited: {last_error}). "
            "Try wikipedia_search, a shorter query, or visit a likely URL directly.")


@tool
def wikipedia_search(query: str) -> str:
    """Search English Wikipedia and return matching article titles, URLs and snippets.
    Use this to find the right Wikipedia page, then open it with visit_webpage."""
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 6},
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    hits = r.json().get("query", {}).get("search", [])
    if not hits:
        return "No Wikipedia results."
    return "\n\n".join(
        f"[{h['title']}](https://en.wikipedia.org/wiki/{h['title'].replace(' ', '_')})\n"
        + re.sub(r"<[^>]+>", "", h.get("snippet", ""))
        for h in hits
    )


@tool
def visit_webpage(url: str) -> str:
    """Visit a webpage and return its main content as markdown text."""
    from bs4 import BeautifulSoup
    from markdownify import markdownify
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()
    except Exception as e:
        return f"Error fetching the webpage: {e}"
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup.find(id="mw-content-text") or soup.body or soup
    text = re.sub(r"\n{3,}", "\n\n", markdownify(str(main)).strip())
    limit = 40000
    return text[:limit] + ("\n\n..._content truncated_..." if len(text) > limit else "")


# ---------------------------------------------------------------- files & media

@tool
def read_table_file(file_path: str) -> str:
    """Read an Excel (.xlsx) or CSV file and return its full contents as text."""
    if file_path.endswith(".csv"):
        sheets = {"csv": pd.read_csv(file_path)}
    else:
        sheets = pd.read_excel(file_path, sheet_name=None)
    return "\n\n".join(f"Sheet: {name}\n{df.to_string()}" for name, df in sheets.items())


@tool
def read_text_file(file_path: str) -> str:
    """Read a text or code file (.txt, .py, .json, .md) and return its contents."""
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        return f.read()


@tool
def analyze_media_file(file_path: str, question: str) -> str:
    """Listen to an audio file or look at an image/PDF and answer a question about it.
    Use for .mp3, .wav, .png, .jpg, .pdf files. Example question: 'Transcribe this audio word for word.'"""
    uploaded = _client.files.upload(file=file_path)
    return _gemini_generate([uploaded, question])


@tool
def analyze_youtube_video(url: str, question: str) -> str:
    """Watch a public YouTube video and answer a question about it."""
    return _gemini_generate([types.Part.from_uri(file_uri=url, mime_type="video/*"), question])


# ---------------------------------------------------------------- code execution

@tool
def run_python(code: str) -> str:
    """Execute Python code and return what it prints. Use for math, counting, sorting,
    and data processing (pandas is available). Always print() the results you need."""
    import subprocess
    import sys
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "Error: code timed out after 60 seconds."
    output = (r.stdout + (f"\nSTDERR:\n{r.stderr}" if r.stderr else "")).strip()
    return output[:20000] or "(no output; remember to print() results)"


ALL_TOOLS = [
    web_search, wikipedia_search, visit_webpage,
    read_table_file, read_text_file, analyze_media_file, analyze_youtube_video,
    run_python,
]