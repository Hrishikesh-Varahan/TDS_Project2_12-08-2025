import os
import json
import requests

# Get AI Pipe API key from environment variable
api_key = os.getenv("AIPIPE_API_KEY")
if not api_key:
    raise ValueError("AIPIPE_API_KEY environment variable is not set.")

# AI Pipe API configuration
BASE_URL = "https://api.aipipe.ai/v1"  # Replace if AI Pipe has a different base URL
MODEL_NAME = "gpt-4o-mini"  # Change to the exact AI Pipe model you want to use

SYSTEM_PROMPT = """
You are a data extraction and analysis assistant.  
Your job is to:
1. Write Python code that scrapes the relevant data needed to answer the user's query. If no URL is given, then see "uploads" folder and read the files provided there and give relevant metadata.
2. List all Python libraries that need to be installed for your code to run.
3. Identify and output the main questions that the user is asking, so they can be answered after the data is scraped.

You must respond **only** in valid JSON following the given schema:
{
  "code": "string — Python scraping code as plain text",
  "libraries": ["string — names of required libraries"],
  "questions": ["string — extracted questions"]
}
Do not include explanations, comments, or extra text outside the JSON.
"""

async def parse_question_with_llm(question_text, uploaded_files=None, urls=None, folder="uploads"):
    uploaded_files = uploaded_files or []
    urls = urls or []

    user_prompt = f"""
Question:
"{question_text}"

Uploaded files:
"{uploaded_files}"

URLs:
"{urls}"

You are a data extraction specialist.
Your task is to generate Python 3 code that loads, scrapes, or reads the data needed to answer the user's question.

1(a). Always store the final dataset in a file as {folder}/data.csv file. And if you need to store other files then also store them in this folder. Lastly, add the path and a brief description about the file in "{folder}/metadata.txt".
1(b). Create code to collect metadata about the data that you collected from scraping (eg. storing details of df using df.info, df.columns, df.head() etc.) in a "{folder}/metadata.txt" file that will help other model to generate code. Add code for creating any folder that doesn't exist like "{folder}".

2. Do not perform any analysis or answer the question. Only write code to collect or add metadata.

3. The code must be self-contained and runnable without manual edits.

4. Use only Python standard libraries plus pandas, numpy, beautifulsoup4, and requests unless otherwise necessary.

5. If the data source is a webpage, download and parse it. If it’s a CSV/Excel, read it directly.

6. Do not explain the code.

7. Output only valid Python code.

8. Just scrap the data don’t do anything fancy.

Return a JSON with:
1. The 'code' field — Python code that answers the question.
2. The 'libraries' field — list of required pip install packages.
3. Don't add libraries that came installed with python like io.
4. Your output will be executed inside a Python REPL.
5. Don't add comments

Only return JSON like:
{{
  "code": "<...>",
  "libraries": ["pandas", "matplotlib"],
  "questions": ["..."]
}}

lastly i am saying again don't try to solve these questions.
in metadata also add JSON answer format if present.
"""

    # API request to AI Pipe
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0
        }
    )

    if response.status_code != 200:
        raise RuntimeError(f"AI Pipe API error: {response.text}")

    result_text = response.json()["choices"][0]["message"]["content"]

    # Ensure uploads folder exists
    file_path = os.path.join(folder, "metadata.txt")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")

    return json.loads(result_text)


SYSTEM_PROMPT2 = """
You are a data analysis assistant.  
Your job is to:
1. Write Python code to solve these questions with provided metadata.
2. List all Python libraries that need to be installed for the code to run.
3. Also add code to save the result to "{folder}/result.json" or any filetype you find suitable (eg. save img files like "{folder}/img.png").

Do not include explanations, comments, or extra text outside the JSON.
"""

async def answer_with_data(question_text, folder="uploads"):
    metadata_path = os.path.join(folder, "metadata.txt")
    with open(metadata_path, "r") as file:
        metadata = file.read()

    user_prompt = f"""
Question:
{question_text}

metadata:
{metadata}

Return a JSON with:
1. The 'code' field — Python code that answers the question.
2. The 'libraries' field — list of required pip install packages.
3. Don't add libraries that came installed with python like "io".
4. Your output will be executed inside a Python REPL.
5. Don't add comments
6. Convert any image/visualisation if present, into base64 PNG and add it to the result.

You must respond **only** in valid JSON with these properties:

  "code": "string — Python scraping code as plain text",
  "libraries": ["string — names of required libraries"]

lastly follow answer format and save answer of questions in result as JSON file.
"""

    file_path = os.path.join(folder, "result.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")

    # API request to AI Pipe
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT2.format(folder=folder)},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0
        }
    )

    if response.status_code != 200:
        raise RuntimeError(f"AI Pipe API error: {response.text}")

    return json.loads(response.json()["choices"][0]["message"]["content"])
