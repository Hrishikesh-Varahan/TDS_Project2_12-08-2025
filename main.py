import os
import json
import httpx
import logging

AI_PIPE_URL = os.getenv("AI_PIPE_URL", "https://api.ai-pipe.com/v1/generate")
AI_PIPE_KEY = os.getenv("AI_PIPE_KEY")

logger = logging.getLogger(__name__)

async def parse_question_with_llm(question_text, uploaded_files, folder):
    """
    Sends the question and file info to AI Pipe API to get scraping code + metadata.
    """
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
    """
    Sends the processed CSV and the user's follow-up questions to AI Pipe API to get analysis code.
    """
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
