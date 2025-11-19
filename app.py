# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from model_runner import build_response
import uvicorn
from datetime import datetime

app = FastAPI(title="Intent Agent")

class PredictRequest(BaseModel):
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    text: str
    metadata: Optional[Dict[str, Any]] = None

@app.get("/health")
async def health():
    return {"status":"ok", "ready": True}

@app.post("/predict")
async def predict(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    resp = build_response(req.request_id or "r-"+datetime.utcnow().isoformat(), req.text)
    # attach session/metadata if provided
    if req.session_id:
        resp["session_id"] = req.session_id
    if req.metadata:
        resp["metadata"] = req.metadata
    return resp

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
