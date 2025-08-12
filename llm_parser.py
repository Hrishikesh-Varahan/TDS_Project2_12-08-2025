import os
import json
import uuid
import shutil
import logging
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
import httpx

# -----------------------------
# Config
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

API_URL = os.getenv("API_URL", "https://aipipe.org/openrouter/v1/chat/completions")
API_KEY = os.getenv("AIPIPE_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")  # Change to your desired model

if not API_KEY:
    raise ValueError("Missing API key. Set AIPIPE_TOKEN in Render environment variables.")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# Step 1: LLM call to extract metadata
# -----------------------------
async def parse_question_with_llm(file_path: str):
    system_prompt = (
        "You are a data analysis assistant. "
        "Read the uploaded CSV file's name and infer relevant metadata. "
        "Return JSON only in this format: "
        '{"columns": ["..."], "data_summary": "..."} '
        "Do not add explanations or text outside JSON."
    )

    user_prompt = f"The file path is: {file_path}. Please return only valid JSON."

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        content = response.json()

        llm_response = content["choices"][0]["message"]["content"]
        try:
            return json.loads(llm_response)
        except json.JSONDecodeError:
            logging.error("Invalid JSON from LLM. Raw output: %s", llm_response)
            raise ValueError("LLM did not return valid JSON")

# -----------------------------
# Step 2: LLM call to analyze data
# -----------------------------
async def answer_with_data(metadata: dict):
    system_prompt = (
        "You are a skilled data analyst. "
        "Using the provided metadata, write a short analysis summary. "
        "Return JSON only in this format: "
        '{"insight": "..."}'
    )

    user_prompt = f"Metadata: {json.dumps(metadata)}"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        content = response.json()

        llm_response = content["choices"][0]["message"]["content"]
        try:
            return json.loads(llm_response)
        except json.JSONDecodeError:
            logging.error("Invalid JSON from LLM. Raw output: %s", llm_response)
            raise ValueError("LLM did not return valid JSON")

# -----------------------------
# Upload route
# -----------------------------
@app.post("/api")
async def upload_file(file: UploadFile = File(...)):
    # Create unique folder for upload
    folder_id = str(uuid.uuid4())
    folder_path = os.path.join(UPLOAD_DIR, folder_id)
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logging.info(f"Step-1: File uploaded to {file_path}")

    # Step-2: Get metadata from LLM
    metadata = await parse_question_with_llm(file_path)
    logging.info(f"Step-2: Metadata: {metadata}")

    # Step-3: Get analysis from LLM
    analysis = await answer_with_data(metadata)
    logging.info(f"Step-3: Analysis: {analysis}")

    return {"metadata": metadata, "analysis": analysis}

# -----------------------------
# Root HTML
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <body>
        <h2>Upload Files to FastAPI</h2>
        <form action="/api" enctype="multipart/form-data" method="post">
            <input name="file" type="file">
            <input type="submit" value="Submit">
        </form>
    </body>
    </html>
    """
