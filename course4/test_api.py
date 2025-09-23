import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture
def valid_payload():
    return {
        "age": 39,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }


def test_get_data(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "message": "Welcome to the MLops Project 3 API Created with FastAPI!"
    }


def test_predict_returns_label(client, valid_payload):
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body

    assert body["prediction"] in ["<=50K", ">50K", "<=50K", ">50K"]


def test_predict_missing_field_returns_422(client, valid_payload):
    bad_payload = valid_payload.copy()
    bad_payload.pop("native-country")
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422

    assert "detail" in resp.json()
