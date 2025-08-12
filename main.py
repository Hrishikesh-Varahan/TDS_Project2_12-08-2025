import os
import json
import httpx
import logging
from fastapi import FastAPI, UploadFile, Form
from typing import Dict

# Environment variables
AI_PIPE_URL = os.getenv("AI_PIPE_URL", "https://api.ai-pipe.com/v1/generate")
AI_PIPE_KEY = os.getenv("AI_PIPE_KEY")

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Pipe API Wrapper")

# ------------------------
# Core Functions
# ------------------------
async def parse_question_with_llm(question_text, uploaded_files, folder):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AI_PIPE_KEY}"
    }

    prompt = f"""
    You are a Python data scraping assistant. 
    The user provided the following question:
    {question_text}

    Uploaded files:
    {list(uploaded_files.keys())}

    Generate valid JSON in this format:
    {{
        "code": "Python scraping code as string",
        "libraries": ["list", "of", "libraries"],
        "questions": "Follow-up question for answering phase"
    }}
    """

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(AI_PIPE_URL, headers=headers, json={"prompt": prompt})
        resp.raise_for_status()
        try:
            return json.loads(resp.text)
        except Exception:
            logger.error(f"Invalid JSON from LLM: {resp.text}")
            raise

async def answer_with_data(questions, folder):
    csv_path = os.path.join(folder, "data.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError("data.csv not found for answering phase")

    with open(csv_path, "r") as f:
        csv_content = f.read()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AI_PIPE_KEY}"
    }

    prompt = f"""
    You are a Python data analysis assistant.
    The following CSV data is available:

    {csv_content[:2000]}  # only send first 2000 chars to avoid overload

    The user has the following analysis questions:
    {questions}

    Generate valid JSON in this format:
    {{
        "code": "Python code to answer the question",
        "libraries": ["list", "of", "libraries"]
    }}
    """

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(AI_PIPE_URL, headers=headers, json={"prompt": prompt})
        resp.raise_for_status()
        try:
            return json.loads(resp.text)
        except Exception:
            logger.error(f"Invalid JSON from LLM: {resp.text}")
            raise

# ------------------------
# API Endpoints
# ------------------------

@app.get("/")
def root():
    return {"message": "AI Pipe API is running"}

@app.post("/parse")
async def parse_endpoint(question: str = Form(...), file: UploadFile = None):
    folder = "./uploads"
    os.makedirs(folder, exist_ok=True)

    uploaded_files: Dict[str, str] = {}
    if file:
        file_path = os.path.join(folder, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        uploaded_files[file.filename] = file_path

    result = await parse_question_with_llm(question, uploaded_files, folder)
    return result

@app.post("/answer")
async def answer_endpoint(questions: str = Form(...)):
    folder = "./uploads"
    result = await answer_with_data(questions, folder)
    return result
