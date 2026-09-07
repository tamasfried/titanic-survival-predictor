"""
FastAPI backend for the Titanic survival predictor.

Loads the trained scikit-learn pipeline (produced by training/train.py) once
at startup, then exposes:
  - GET  /health   simple check that the server + model are up
  - POST /predict  takes passenger details, returns a survival prediction

Run it with:
    uvicorn main:app --reload
"""

import pathlib

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = pathlib.Path(__file__).resolve().parent / "model" / "titanic_model.joblib"

app = FastAPI(title="Titanic Survival Predictor")

# Allow the simple HTML/JS frontend (opened directly from disk, or served
# from a different port) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Run `python train.py` in the training/ folder first."
        )
    model = joblib.load(MODEL_PATH)


class PassengerDetails(BaseModel):
    pclass: int = Field(..., ge=1, le=3, description="Ticket class: 1 = 1st, 2 = 2nd, 3 = 3rd")
    sex: str = Field(..., description="'male' or 'female'")
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sibsp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    fare: float = Field(..., ge=0, description="Fare paid")
    embarked: str = Field(..., description="Port of embarkation: 'C', 'Q', or 'S'")


class PredictionResponse(BaseModel):
    survived: bool
    probability: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: PassengerDetails):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    if passenger.sex not in ("male", "female"):
        raise HTTPException(status_code=422, detail="sex must be 'male' or 'female'")
    if passenger.embarked not in ("C", "Q", "S"):
        raise HTTPException(status_code=422, detail="embarked must be 'C', 'Q', or 'S'")

    row = pd.DataFrame([passenger.dict()])
    probability_survived = float(model.predict_proba(row)[0][1])
    survived = probability_survived >= 0.5

    return PredictionResponse(survived=survived, probability=probability_survived)
