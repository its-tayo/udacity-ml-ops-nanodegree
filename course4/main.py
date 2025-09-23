import os
import pickle
import subprocess
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from utils.model import process_data
from utils.model import inference

app = FastAPI()

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
MODEL_DIR = HERE / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
LB_PATH = MODEL_DIR / "label_binarizer.pkl"


def _ensure_artifacts():
    missing = [
        p for p in (MODEL_PATH, ENCODER_PATH, LB_PATH)
        if not p.exists()
    ]
    if not missing:
        return

    # Only auto-pull if allowed. Set DVC_AUTO_PULL=0 to disable.
    if os.getenv("DVC_AUTO_PULL", "1") != "1":
        raise FileNotFoundError(
            f"Missing artifacts: {', '.join(map(str, missing))}. "
            "Run `dvc pull -R course4/model` or set DVC_AUTO_PULL=1."
        )

    # Try pulling everything under course4/model from the repo root.
    try:
        subprocess.run(
            ["dvc", "pull", "-R", "course4/model"],
            check=True,
            cwd=REPO_ROOT
        )
    except FileNotFoundError:
        raise RuntimeError(
            "DVC is not installed. Install `dvc[s3]` or disable auto-pull"
            "with DVC_AUTO_PULL=0."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"`dvc pull` failed: {e}")

    # Verify again
    still_missing = [
        p for p in (MODEL_PATH, ENCODER_PATH, LB_PATH)
        if not p.exists()
    ]
    if still_missing:
        raise FileNotFoundError(
            f"Artifacts still missing after `dvc pull`: "
            f"{', '.join(map(str, still_missing))}"
        )


_ensure_artifacts()


with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(ENCODER_PATH, "rb") as encoder_file:
    encoder = pickle.load(encoder_file)

with open(LB_PATH, "rb") as lb_file:
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
