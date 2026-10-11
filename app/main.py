"""Churn prediction API."""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from .schemas import BatchRequest, BatchResponse, CustomerFeatures, Prediction

logger = logging.getLogger("churn-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model/churn_model.joblib"))
THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.metrics = None, {}
    try:
        app.state.model = joblib.load(MODEL_PATH)
        metrics_file = MODEL_PATH.with_name("metrics.json")
        if metrics_file.exists():
            app.state.metrics = json.loads(metrics_file.read_text())
        logger.info("Loaded model from %s", MODEL_PATH)
    except Exception:
        logger.exception("Could not load model from %s", MODEL_PATH)
    yield


app = FastAPI(title="Customer Churn API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{ms:.2f}"
    logger.info("%s %s -> %s in %.1f ms", request.method, request.url.path, response.status_code, ms)
    return response


def _model(request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def _risk(p: float) -> str:
    return "low" if p < 0.3 else "medium" if p < 0.6 else "high"


def _predict(model, customers) -> list:
    df = pd.DataFrame([c.model_dump() for c in customers])
    probs = model.predict_proba(df)[:, 1]
    return [
        Prediction(churn_probability=round(float(p), 4),
                   prediction="Churn" if p >= THRESHOLD else "Stay",
                   risk_level=_risk(float(p)))
        for p in probs
    ]


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": request.app.state.model is not None}


@app.get("/model-info")
def model_info(request: Request):
    _model(request)
    return {**request.app.state.metrics, "threshold": THRESHOLD}


@app.post("/predict", response_model=Prediction)
def predict(customer: CustomerFeatures, request: Request):
    return _predict(_model(request), [customer])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request):
    return BatchResponse(predictions=_predict(_model(request), batch.customers))
