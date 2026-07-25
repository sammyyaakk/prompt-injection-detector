from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict

app = FastAPI(title="Prompt Injection Detector")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    model: str = "embedding"


class PredictResponse(BaseModel):
    text: str
    model: str
    label: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    if request.model not in ("embedding", "tfidf"):
        raise HTTPException(status_code=400, detail="model must be 'embedding' or 'tfidf'")

    return predict(request.text, model=request.model)