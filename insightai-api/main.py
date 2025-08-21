import os
import io
from typing import Optional
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="InsightAI")

class SummaryRequest(BaseModel):
    text: str
    topic: Optional[str] = "general"
    model: Optional[str] = "gpt-4o-mini"

@app.get('/')
def root():
    return {"ok": True, "name": "InsightAI API"}

@app.post('/extract')
async def extract(file: UploadFile = File(...)):
    content = await file.read()
    text = ""
    if file.filename.lower().endswith('.pdf'):
        try:
            text = extract_text(io.BytesIO(content))
        except Exception:
            # fallback with PyMuPDF
            doc = fitz.open(stream=content, filetype='pdf')
            text = "\n".join([page.get_text() for page in doc])
    else:
        text = content.decode('utf-8', errors='ignore')
    return {"chars": len(text), "text": text[:2000]}

SYSTEM_PROMPT = "You are a precise analyst. Given TEXT: \n- Produce JSON with keys: executive_summary[], insights[], decisions[], actions[{title,owner,due}], risks[].\n- Keep items concise and specific."

async def call_openai(prompt: str, model: str):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # dry-run stub
        return {
            "executive_summary": ["(stub) Provide API key to enable real LLM."],
            "insights": ["Example insight"],
            "decisions": [],
            "actions": [{"title":"Follow up","owner":"You","due":"Next week"}],
            "risks": ["Data missing"]
        }
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": prompt}
        ],
        "temperature": 0.2
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        import json as _json
        try:
            return _json.loads(text)
        except Exception:
            return {"raw": text}

@app.post('/summarize')
async def summarize(req: SummaryRequest):
    prompt = f"Analyze the following text and return structured JSON:\n---\n{req.text[:12000]}\n---\nTopic: {req.topic}\n"
    result = await call_openai(prompt, req.model or "gpt-4o-mini")
    return result
