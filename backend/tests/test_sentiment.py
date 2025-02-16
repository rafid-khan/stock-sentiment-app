from fastapi.testclient import TestClient
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app

@pytest.fixture(scope="module")
def client():
    """Fixture to create a FastAPI test client."""
    return TestClient(app)

def test_root(client):
    """Test root API endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "FastAPI is running successfully"}

def test_predict_sentiment(client):
    """Test sentiment prediction with valid input."""
    response = client.post("/predict/", json={
        "ticker": "AAPL",
        "headlines": ["Apple releases new iPhone", "Apple stock rises"]
    })
    assert response.status_code == 200
    json_data = response.json()
    
    assert "sentiment_score" in json_data
    assert isinstance(json_data["sentiment_score"], float)
    assert "sentiments" in json_data
    assert isinstance(json_data["sentiments"], list)
    assert all(isinstance(score, float) for score in json_data["sentiments"])

def test_predict_no_headlines(client):
    """Test API returns 400 error when no headlines are provided."""
    response = client.post("/predict/", json={"ticker": "AAPL", "headlines": []})
    assert response.status_code == 400
    assert response.json() == {"detail": "No headlines provided."}

def test_predict_missing_ticker(client):
    """Test API handles missing ticker field properly."""
    response = client.post("/predict/", json={"headlines": ["Apple launches new product"]})
    assert response.status_code == 422  # FastAPI automatically returns 422 for validation errors

def test_predict_invalid_headlines(client):
    """Test API rejects incorrect data types (headlines should be a list)."""
    response = client.post("/predict/", json={"ticker": "AAPL", "headlines": "Invalid headline"})
    assert response.status_code == 422

def test_predict_extreme_sentiment(client):
    """Test API handles extreme sentiment cases."""
    response = client.post("/predict/", json={
        "ticker": "TSLA",
        "headlines": ["Tesla stock crashes after CEO resigns!", "Tesla is the future of AI and technology!"]
    })
    assert response.status_code == 200
    json_data = response.json()
    
    assert json_data["sentiment_score"] <= 1
    assert json_data["sentiment_score"] >= -1
