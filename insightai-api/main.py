from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SummarizeInput(BaseModel):
    text: str
    topics: list[str] | None = None

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
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
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
