import requests

from agent import GaiaAgent
from utils import API_URL, download_task_file

TASK_ID = "8e867cd7-cff9-4e6c-867a-ff5ddc2550be"   # Mercedes Sosa (no file); set None for random

if TASK_ID:
    all_questions = requests.get(f"{API_URL}/questions", timeout=30).json()
    q = next(x for x in all_questions if x["task_id"] == TASK_ID)
else:
    q = requests.get(f"{API_URL}/random-question", timeout=30).json()

print("TASK:", q["task_id"])
print("QUESTION:", q["question"])
print("FILE:", q.get("file_name") or "none")

file_path = download_task_file(q["task_id"], q.get("file_name", ""))
print("\nANSWER:", GaiaAgent(session_id="local-test")(q["question"], file_path, q["task_id"]))
