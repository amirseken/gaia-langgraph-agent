import os
import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import ALL_TOOLS, GEMINI_MODELS

load_dotenv()

# Tracing is optional: enabled only when Langfuse keys are set, so the app still runs without them.
LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
if LANGFUSE_ENABLED:
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler


SYSTEM_PROMPT = """You are answering questions from the GAIA benchmark. Answers are graded by EXACT MATCH.

How to work:
- Use your tools. Never guess or assume what a file, webpage, video or recording contains: call a tool and read its output first.
- If a file path is given, open it with the right tool: analyze_media_file for audio/images/PDF, read_table_file for Excel/CSV, read_text_file for text/code.
- For math, counting, sorting or data processing, use run_python and print the result rather than computing in your head.
- If web_search returns nothing, try wikipedia_search, a shorter query, or visit a likely URL directly.
- Open pages rather than trusting search snippets. Prefer authoritative sources.
- As soon as you have enough evidence, stop and answer. Do not re-check information you already confirmed.
- If an approach fails twice (errors, empty results), switch approaches or answer with the best evidence you have. Do not spend many steps on a side issue.
- Do not install software. Use only the tools and libraries already available.

When you answer (with no tool call), give ONLY the final answer:
- No explanation, no full sentence, no "The answer is".
- Numbers: digits only, no thousands separators, no units unless asked.
- Strings: no articles, no abbreviations unless the question uses them.
- Lists: comma-separated with one space after each comma. Follow any ordering the question requests.
- Follow every formatting instruction in the question exactly."""


# ---------------------------------------------------------------- state

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # conversation; new messages are appended
    question: str
    answer: str


# ---------------------------------------------------------------- models

def make_llm(model: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
        max_retries=1,   # fail fast; the fallback chain handles switching models
        timeout=120,
    )


_models = [make_llm(m) for m in GEMINI_MODELS]

# Agent model: tools attached, strongest first, falls back down the list on errors.
agent_llm = _models[0].bind_tools(ALL_TOOLS).with_fallbacks(
    [m.bind_tools(ALL_TOOLS) for m in _models[1:]]
)

# Formatting model: no tools; cheapest first since the task is easy.
format_llm = _models[-1].with_fallbacks(_models[:-1])

answer_llm = _models[0].with_fallbacks(_models[1:])

def text_of(message) -> str:
    """Gemini may return content as a string or as a list of parts; return plain text."""
    content = message.content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content or ""


def clean_answer(raw: str) -> str:
    text = str(raw).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        text = lines[-1]
    text = re.sub(r"^(final answer|answer|the answer is)\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("`*").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    text = text.rstrip(".")
    if "," in text:
        text = ", ".join(p.strip() for p in text.split(","))
    return text


# ---------------------------------------------------------------- nodes

def agent_node(state: AgentState) -> dict:
    response = agent_llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    for call in getattr(response, "tool_calls", []) or []:
        print(f"  -> {call['name']}({str(call['args'])[:150]})")
    return {"messages": [response]}


def format_answer_node(state: AgentState) -> dict:
    raw = text_of(state["messages"][-1])
    prompt = (
        "Extract the final answer from the response below, formatted exactly as the question requires.\n"
        "Output ONLY the answer itself: no sentence, no explanation, no markdown, no quotes.\n\n"
        f"QUESTION:\n{state['question']}\n\nRESPONSE:\n{raw}"
    )
    try:
        formatted = clean_answer(text_of(format_llm.invoke(prompt)))
    except Exception as e:
        print(f"  (format pass failed: {e})")
        formatted = ""
    answer = formatted or clean_answer(raw)  # never let the format pass erase a real answer
    print(f"  RAW: {raw[:200]!r}\n  FORMATTED: {answer!r}")
    return {"answer": answer}


# ---------------------------------------------------------------- graph

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(ALL_TOOLS))
builder.add_node("format_answer", format_answer_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "format_answer"})
builder.add_edge("tools", "agent")
builder.add_edge("format_answer", END)

graph = builder.compile()


# ---------------------------------------------------------------- public interface
class GaiaAgent:
    """Same interface as the smolagents BasicAgent: agent(question, file_path) -> answer."""

    def __init__(self, session_id: str | None = None):
        # One session per evaluation run, so all 20 questions are grouped in Langfuse.
        self.session_id = session_id

    def _config(self, task_id: str | None = None) -> dict:
        config = {"recursion_limit": 25, "run_name": "gaia-question"}
        if LANGFUSE_ENABLED:
            config["callbacks"] = [CallbackHandler()]
            metadata = {"langfuse_tags": ["langgraph", "gaia"]}
            if self.session_id:
                metadata["langfuse_session_id"] = self.session_id
            if task_id:
                metadata["task_id"] = task_id
            config["metadata"] = metadata
        return config

    def __call__(self, question: str, file_path: str | None = None, task_id: str | None = None) -> str:
        task = question
        if file_path:
            task += f"\n\nThe attached file is saved locally at: {file_path}"
        inputs = {"messages": [HumanMessage(content=task)], "question": question, "answer": ""}
        config = self._config(task_id)
        state = None
        try:
            for step_state in graph.stream(inputs, config=config, stream_mode="values"):
                state = step_state  # keep the latest state, also needed if the step limit is hit
            return state["answer"]
        except GraphRecursionError:
            print("  (step limit reached; forcing a best-guess answer)")
            return self._best_guess(question, state, config)
        except Exception as e:
            print(f"  (agent error: {e})")
            return ""
        finally:
            if LANGFUSE_ENABLED:
                get_client().flush()  # send traces now; the Space may restart before background sending

    def _best_guess(self, question: str, state, config: dict) -> str:
        if not state:
            return ""
        transcript = "\n\n".join(text_of(m)[:3000] for m in state["messages"][-12:] if text_of(m))
        prompt = (
            "You ran out of research steps. Based ONLY on the evidence gathered below, give your best "
            "final answer to the question. Output ONLY the answer, formatted as the question requires. "
            "If the evidence contains no answer at all, output nothing.\n\n"
            f"QUESTION:\n{question}\n\nEVIDENCE:\n{transcript}"
        )
        try:
            callbacks = {"callbacks": config["callbacks"]} if "callbacks" in config else {}
            return clean_answer(text_of(answer_llm.invoke(prompt, config=callbacks)))
        except Exception as e:
            print(f"  (best-guess failed: {e})")
            return ""
