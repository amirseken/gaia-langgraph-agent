import os

import requests
from huggingface_hub import hf_hub_download

API_URL = "https://agents-course-unit4-scoring.hf.space"


def download_task_file(task_id: str, file_name: str) -> str | None:
    if not file_name:
        return None
    os.makedirs("gaia_files", exist_ok=True)
    path = os.path.join("gaia_files", file_name)
    if os.path.exists(path):
        return path
    try:
        r = requests.get(f"{API_URL}/files/{task_id}", timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    except Exception as e:
        print(f"Course API file download failed ({e}); trying GAIA dataset...")
    try:
        return hf_hub_download(
            repo_id="gaia-benchmark/GAIA", repo_type="dataset",
            filename=f"2023/validation/{file_name}", local_dir="gaia_files",
            token=os.getenv("HF_TOKEN"),
        )
    except Exception as e:
        print(f"GAIA dataset download failed: {e}")
        return None
