import pickle
from pathlib import Path

import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.data import process_data
from utils.model import train_model, compute_model_metrics, inference

BASE_DIR = Path(__file__).resolve().parent


@pytest.fixture
def df():
    return pd.read_csv(BASE_DIR / "data" / "census.csv")


@pytest.fixture
def train_data(df):
    cat_features = (
        df.drop("salary", axis=1).select_dtypes("object").columns
    )
    x_train, y_train, _, _ = process_data(
        df, categorical_features=cat_features, label="salary"
    )
    return x_train, y_train


@pytest.fixture
def model():
    with open(BASE_DIR / "model" / "model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


@pytest.fixture
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


@pytest.fixture
def split_data(df):
    train, test = train_test_split(df, test_size=0.20)
    return train, test


def test_process_data_training(split_data, cat_features):
    train, _ = split_data
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    assert X_train.shape[0] == y_train.shape[0]
    assert encoder is not None
    assert lb is not None


def test_compute_model_metrics(train_data):
    x_train, y_train = train_data[0], train_data[1]
    model = train_model(x_train, y_train)
    y_preds = model.predict(x_train)
    precision, recall, fbeta = compute_model_metrics(y_train, y_preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_inference(model, train_data):
    x_data, y_data = train_data[0], train_data[1]
    y_preds = inference(model, x_data)
    assert y_preds is not None
    assert len(y_preds) == len(y_data)
