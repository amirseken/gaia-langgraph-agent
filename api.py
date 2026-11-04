import os
import shutil
import tempfile
import time

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from agent import GaiaAgent
from utils import download_task_file

app = FastAPI(
    title="GAIA Agent API",
    version="1.0.0",
    description="A LangGraph + Gemini agent that answers questions using web, file, audio and video tools.",
)

# Created once at startup and reused for every request.
agent = GaiaAgent(session_id="api")


# ---------------------------------------------------------------- auth

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Depends(api_key_header)) -> None:
    """If API_KEY is set, every request must send it in the X-API-Key header.
    Each request spends model quota, so a public endpoint should not be open to everyone."""
    expected = os.getenv("API_KEY")
    if expected and key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------- schemas

class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000, description="The question to answer")
    task_id: str | None = Field(None, description="GAIA task ID, used to download an attached file")
    file_name: str | None = Field(None, description="GAIA file name for the task, if it has one")


class AnswerResponse(BaseModel):
    answer: str
    duration_seconds: float


# ---------------------------------------------------------------- endpoints


@app.get("/hello")
def hello():
    return {"message": "Hello!"}
    
@app.get("/health")
def health() -> dict:
    """Lightweight check that the service is running (used by Docker and hosting platforms)."""
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse, dependencies=[Depends(require_api_key)])
def answer(request: AnswerRequest) -> AnswerResponse:
    """Answer a question, optionally with a GAIA task file."""
    start = time.time()
    file_path = None
    if request.task_id and request.file_name:
        file_path = download_task_file(request.task_id, request.file_name)
        if file_path is None:
            raise HTTPException(status_code=404, detail="Could not download the task file")
    result = agent(request.question, file_path, request.task_id)
    return AnswerResponse(answer=result, duration_seconds=round(time.time() - start, 2))


@app.post("/answer/upload", response_model=AnswerResponse, dependencies=[Depends(require_api_key)])
def answer_with_upload(question: str = Form(...), file: UploadFile = File(...)) -> AnswerResponse:
    """Answer a question about an uploaded file (Excel, CSV, audio, image, PDF, code)."""
    start = time.time()
    suffix = os.path.splitext(file.filename or "")[1]  # keep the extension so tools can detect the type
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name
    try:
        result = agent(question, path)
    finally:
        os.remove(path)  # never leave uploaded files on the server
    return AnswerResponse(answer=result, duration_seconds=round(time.time() - start, 2))