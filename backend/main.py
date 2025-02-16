from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Union

app = FastAPI()

# CORS Middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

analyzer = SentimentIntensityAnalyzer()

@app.get("/")
def read_root() -> Dict[str, str]:
    """Root endpoint to verify API is running."""
    return {"message": "FastAPI is running successfully"}

class StockRequest(BaseModel):
    """Request model for sentiment analysis."""
    ticker: str
    headlines: List[str]

def predict_sentiment(text: str) -> float:
    """
    Predict sentiment score for a given text using VADER.

    Args:
        text (str): News headline or text input.

    Returns:
        float: Sentiment score ranging from -1 (negative) to 1 (positive).
    """
    score = analyzer.polarity_scores(text)
    return score["compound"]

# with open("models/logistic_regression_stock_sentiment.pk1", "rb") as f:
#     model = pickle.load(f)
# with open("models/tfidf_vectorizer.pk1", "rb") as f:
#     vectorizer = pickle.load(f)

# def predict_sentiment(sentence: str) -> float:
#     """Use the trained ML model to predict sentiment."""
#     sentence = sentence.lower()
#     sentence = re.sub(r'[^\w\s]', '', sentence)  
#     vectorized_text = vectorizer.transform([sentence])
#     proba = model.predict_proba(vectorized_text)[0]
#     sentiment_score = proba[2] - proba[0]
#     return sentiment_score

# The ML model is included for future improvements, but for real-time performance, we use VADER.

def normalize_sentiment_scores(sentiment_scores: List[float]) -> List[float]:
    """
    Normalize sentiment scores between -1 and 1 using MinMaxScaler.

    Args:
        sentiment_scores (List[float]): List of raw sentiment scores.

    Returns:
        List[float]: Normalized sentiment scores.
    """
    if len(sentiment_scores) == 0:
        return sentiment_scores

    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    sentiment_scores = np.array(sentiment_scores).reshape(-1, 1)
    normalized_scores = scaler.fit_transform(sentiment_scores).flatten()
    return list(normalized_scores)

def apply_time_decay(sentiment_scores: List[float], decay_rate: float = 0.9) -> float:
    """
    Apply time decay to sentiment scores, giving more importance to recent headlines.

    Args:
        sentiment_scores (List[float]): List of sentiment scores.
        decay_rate (float, optional): Weight decay factor (default is 0.9).

    Returns:
        float: Weighted sentiment score.
    """
    n = len(sentiment_scores)
    if n == 0:
        return 0 

    weights = np.array([decay_rate**i for i in range(n)][::-1])  
    weights /= np.sum(weights)  
    weighted_scores = np.dot(sentiment_scores, weights) 
    return float(weighted_scores)

def adjust_market_bias(sentiment_score: float, baseline: float = 0.1) -> float:
    """
    Adjust sentiment score to account for general market bias (financial news tends to be positive).

    Args:
        sentiment_score (float): Weighted sentiment score.
        baseline (float, optional): Market bias adjustment factor (default is 0.1).

    Returns:
        float: Adjusted sentiment score.
    """
    return sentiment_score - baseline

def final_sentiment_score(weighted_score: float, scaling_factor: float = 5) -> float:
    """
    Apply sigmoid transformation for smoother sentiment score distribution.

    Args:
        weighted_score (float): Adjusted sentiment score.
        scaling_factor (float, optional): Scaling factor for sigmoid transformation (default is 5).

    Returns:
        float: Final sentiment score between -1 and 1.
    """
    return 2 / (1 + np.exp(-weighted_score * scaling_factor)) - 1

@app.post("/predict/")
async def predict_sentiments(request: StockRequest) -> Dict[str, Union[float, List[float]]]:
    """
    Endpoint for sentiment analysis.

    Receives a stock ticker and news headlines, analyzes sentiment, and returns a final sentiment score.

    Args:
        request (StockRequest): Request model containing stock ticker and list of headlines.

    Returns:
        Dict[str, Union[float, List[float]]]: Overall sentiment score and individual headline sentiments.
    """
    print(f"Received request: {request.model_dump()}") 

    if not request.headlines:
        raise HTTPException(status_code=400, detail="No headlines provided.")

    sentiment_scores = [predict_sentiment(headline) for headline in request.headlines]

    if sentiment_scores:
        sentiment_scores = normalize_sentiment_scores(sentiment_scores) 
        weighted_score = apply_time_decay(sentiment_scores)
        adjusted_score = adjust_market_bias(weighted_score)
        final_score = final_sentiment_score(adjusted_score)
    else:
        final_score = 0.0  

    return {"sentiment_score": final_score, "sentiments": sentiment_scores}
