# InsightAI — Document Summarizer & Action Extractor

Upload PDFs/notes and get executive summaries, insights, and action items.

## Quick Start (API)
```bash
cd insightai-api
cp .env.example .env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 7000
```
OpenAPI docs at `http://localhost:7000/docs`.
