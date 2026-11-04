import os
import sys

# Make the project root importable (agent.py, tools.py, utils.py).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# agent.py and tools.py create API clients at import time. Dummy keys let them
# import without real credentials; no test makes a real API call.
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.pop("LANGFUSE_PUBLIC_KEY", None)  # keep tracing off during tests
os.environ.pop("LANGFUSE_SECRET_KEY", None)