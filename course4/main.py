import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from utils.model import process_data
from utils.model import inference

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

model_filename = BASE_DIR / "model" / "model.pkl"
encoder_filename = BASE_DIR / "model" / "encoder.pkl"
lb_filename = BASE_DIR / "model" / "label_binarizer.pkl"

with open(model_filename, "rb") as model_file:
    model = pickle.load(model_file)

with open(encoder_filename, "rb") as encoder_file:
    encoder = pickle.load(encoder_file)

with open(lb_filename, "rb") as lb_file:
    lb = pickle.load(lb_file)

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class PredictionRequest(BaseModel):
    age: int = Field(..., example=45)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=123456)
    education: str = Field(..., example="Masters")
    education_num: int = Field(..., alias="education-num", example=16)
    marital_status: str = Field(
        ..., alias="marital-status", example="Married-civ-spouse"
    )
    occupation: str = Field(..., example="Exec-managerial")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="Asian-Pac-Islander")
    sex: str = Field(..., example="Female")
    capital_gain: int = Field(..., alias="capital-gain", example=5000)
    capital_loss: int = Field(..., alias="capital-loss", example=200)
    hours_per_week: int = Field(
        ..., alias="hours-per-week", example=50
    )
    native_country: str = Field(
        ..., alias="native-country", example="United-States"
    )


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the MLops Project 3 API Created with FastAPI!"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    df = pd.DataFrame([request.dict(by_alias=True)])
    X, _, _, _ = process_data(
        df,
        categorical_features=categorical_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    pred = inference(model, X)
    label = lb.inverse_transform(pred)[0]

    return {"prediction": label}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
