from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Configure CORS properly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rafid-khan.github.io"],  # Change to "*" if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# ✅ Handle CORS for preflight requests
@app.options("/{full_path:path}")
async def preflight(full_path: str):
    """Handle CORS preflight requests for all routes."""
    response = JSONResponse(content={"message": "Preflight OK"})
    response.headers["Access-Control-Allow-Origin"] = "https://rafid-khan.github.io"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# ✅ Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# ✅ Define request model
class StockRequest(BaseModel):
    ticker: str
    headlines: list[str]

# ✅ Function to get sentiment score using VADER
def predict_sentiment(text):
    score = analyzer.polarity_scores(text)
    return score["compound"]  # Returns a value between -1 and 1

# ✅ Function to normalize sentiment scores
def normalize_sentiment_scores(sentiment_scores):
    if not sentiment_scores:
        return sentiment_scores

    scaler = MinMaxScaler(feature_range=(-1, 1))  
    sentiment_scores = np.array(sentiment_scores).reshape(-1, 1)
    normalized_scores = scaler.fit_transform(sentiment_scores).flatten()
    return list(normalized_scores)

# ✅ Apply time decay (recent headlines matter more)
def apply_time_decay(sentiment_scores, decay_rate=0.9):
    n = len(sentiment_scores)
    if n == 0:
        return 0

    weights = np.array([decay_rate**i for i in range(n)][::-1])  
    weights /= np.sum(weights)  # Normalize weights
    weighted_scores = np.dot(sentiment_scores, weights)
    return weighted_scores

# ✅ Adjust for market bias (finance news is often positive)
def adjust_market_bias(sentiment_score, baseline=0.1):
    return sentiment_score - baseline  

# ✅ Sigmoid transformation for smoother score distribution
def final_sentiment_score(weighted_score, scaling_factor=5):
    return 2 / (1 + np.exp(-weighted_score * scaling_factor)) - 1

# ✅ API Endpoint to Predict Sentiment
@app.post("/predict/")
async def predict_sentiments(request: StockRequest):
    print(f"Received request: {request.dict()}")  

    if not request.headlines:
        return {"sentiment_score": None, "sentiments": []}

    sentiment_scores = [predict_sentiment(headline) for headline in request.headlines]

    if sentiment_scores:
        sentiment_scores = normalize_sentiment_scores(sentiment_scores)
        weighted_score = apply_time_decay(sentiment_scores)
        adjusted_score = adjust_market_bias(weighted_score)
        final_score = final_sentiment_score(adjusted_score)
    else:
        final_score = None  

    # ✅ Ensure CORS headers are present in response
    response = JSONResponse(content={
        "sentiment_score": final_score,
        "sentiments": sentiment_scores
    })
    response.headers["Access-Control-Allow-Origin"] = "https://rafid-khan.github.io"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    
    return response

# ✅ API Endpoint to Get Average Sentiment (Example)
@app.get("/avg_sentiment/")
async def get_avg_sentiment(ticker: str):
    response_data = {
        "avg_sentiment": 0.951851390899973,  
        "date_range": {"start_date": "2025-02-12", "end_date": "2025-02-16"},
        "number_of_articles": 100
    }

    # ✅ Ensure CORS headers are in the response
    response = JSONResponse(content=response_data)
    response.headers["Access-Control-Allow-Origin"] = "https://rafid-khan.github.io"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    
    return response

# ✅ Health Check Endpoint
@app.get("/")
async def health_check():
    return {"status": "API is running"}
