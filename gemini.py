import os
import json
import requests

# Get AI Pipe API key from environment variable
api_key = os.getenv("AIPIPE_API_KEY")
if not api_key:
    raise ValueError("AIPIPE_API_KEY environment variable is not set.")

# AI Pipe API configuration
BASE_URL = os.getenv("AIPIPE_BASE_URL")
MODEL_NAME = "openai/gpt-4o-mini"

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
        BASE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
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
    return result_text


# Example usage
if __name__ == "__main__":
    import asyncio
    output = asyncio.run(parse_question_with_llm(
        "Scrape the latest cricket match scores from ESPN Cricinfo",
        urls=["https://www.espncricinfo.com/"]
    ))
    print(output)
