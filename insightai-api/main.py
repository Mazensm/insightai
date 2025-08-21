from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI(title="InsightAI API", version="0.1.0")


class AnalyzeInput(BaseModel):
    text: str

class SummarizeInput(BaseModel):
    text: str
    topics: list[str] | None = None


def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        # Avoid crashing the whole app if the key is missing
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    return OpenAI(api_key=key)


@app.get("/")
def root():
    return {"message": "InsightAI API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(body: AnalyzeInput):
    text = body.text or ""
    words = len(text.split())
    return {"length": len(text), "words": words}


@app.post("/summarize")
def summarize(body: SummarizeInput):
    topics = body.topics or ["key insights", "risks", "next actions"]
    system = (
        "You are an analyst. Return a concise JSON with fields: "
        "`insights` (bulleted strings), `actions` (numbered steps), "
        "`topics` (dict of topic->summary). Be specific and non-generic."
    )
    prompt = (
        f"Summarize the following content.\n\nTopics to cover: {topics}\n\n"
        f"CONTENT:\n{body.text}\n"
    )

    client = get_client()
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    
    content = "".join(
        block.text.value
        for out in resp.output
        for block in getattr(out, "content", [])
        if getattr(block, "type", "") == "output_text"
    )

    return {"summary": content}
