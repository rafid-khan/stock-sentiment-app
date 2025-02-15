from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler

# Initialize FastAPI app
app = FastAPI()

# ✅ Add CORS Middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (React frontend)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Data model for API request
class StockRequest(BaseModel):
    ticker: str
    headlines: list[str]

# ✅ Function to get sentiment score using VADER
def predict_sentiment(text):
    score = analyzer.polarity_scores(text)
    return score["compound"]  # Returns a value between -1 (negative) and 1 (positive)

# ✅ Function to normalize sentiment scores between -1 and 1
def normalize_sentiment_scores(sentiment_scores):
    if len(sentiment_scores) == 0:
        return sentiment_scores  # Return empty list if no scores

    scaler = MinMaxScaler(feature_range=(-1, 1))  # Scale between -1 and 1
    sentiment_scores = np.array(sentiment_scores).reshape(-1, 1)
    normalized_scores = scaler.fit_transform(sentiment_scores).flatten()
    return list(normalized_scores)

# ✅ Apply time decay (recent headlines are more important)
def apply_time_decay(sentiment_scores, decay_rate=0.9):
    n = len(sentiment_scores)
    if n == 0:
        return 0  # Return 0 if no scores available

    weights = np.array([decay_rate**i for i in range(n)][::-1])  # More weight to recent headlines
    weights /= np.sum(weights)  # Normalize weights
    weighted_scores = np.dot(sentiment_scores, weights)  # Compute weighted sum
    return weighted_scores

# ✅ Adjust for market bias (finance news tends to be positive)
def adjust_market_bias(sentiment_score, baseline=0.1):
    return sentiment_score - baseline  # Adjust for positive-heavy bias

# ✅ Apply sigmoid transformation for smoother score distribution
def final_sentiment_score(weighted_score, scaling_factor=5):
    return 2 / (1 + np.exp(-weighted_score * scaling_factor)) - 1

# ✅ API Endpoint to Predict Sentiment
@app.post("/predict/")
async def predict_sentiments(request: StockRequest):
    print(f"Received request: {request.dict()}")  # Debugging

    if not request.headlines:
        return {"sentiment_score": None, "sentiments": []}

    # Compute sentiment scores for each headline
    sentiment_scores = [predict_sentiment(headline) for headline in request.headlines]

    if sentiment_scores:
        sentiment_scores = normalize_sentiment_scores(sentiment_scores)  # Normalize scores
        weighted_score = apply_time_decay(sentiment_scores)  # Apply time decay
        adjusted_score = adjust_market_bias(weighted_score)  # Correct market bias
        final_score = final_sentiment_score(adjusted_score)  # Apply sigmoid transformation
    else:
        final_score = None  # No headlines = No score

    return {"sentiment_score": final_score, "sentiments": sentiment_scores}
