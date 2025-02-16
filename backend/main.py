from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import requests  # ✅ Added to fetch real news data

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Proper CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rafid-khan.github.io"],  # Change to "*" if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# ✅ Data model for API request
class StockRequest(BaseModel):
    ticker: str
    headlines: list[str]

# ✅ Function to get sentiment score using VADER
def predict_sentiment(text):
    score = analyzer.polarity_scores(text)
    return score["compound"]  # Returns a value between -1 (negative) and 1 (positive)

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
    weights /= np.sum(weights)
    weighted_scores = np.dot(sentiment_scores, weights)
    return weighted_scores

# ✅ Adjust for market bias (finance news is often positive)
def adjust_market_bias(sentiment_score, baseline=0.1):
    return sentiment_score - baseline

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
        sentiment_scores = normalize_sentiment_scores(sentiment_scores)
        weighted_score = apply_time_decay(sentiment_scores)
        adjusted_score = adjust_market_bias(weighted_score)
        final_score = final_sentiment_score(adjusted_score)
    else:
        final_score = None

    return {"sentiment_score": final_score, "sentiments": sentiment_scores}

# ✅ API to Fetch News and Analyze Sentiment
@app.get("/stock_news_sentiment/")
async def get_stock_news_sentiment(ticker: str):
    """
    Fetches stock-related news and analyzes sentiment.
    """
    print(f"Fetching news sentiment for ticker: {ticker}")

    # ✅ Fetch real stock news (Replace with your API key)
    try:
        response = requests.get(f"https://newsapi.org/v2/everything?q={ticker}&apiKey=720f4ab74dmshfd4d338ac93a715p12f4a2jsn00cec9a5efc4")
        news_data = response.json()
    except Exception as e:
        return {"error": f"Failed to fetch news: {str(e)}"}

    # ✅ Extract headlines and analyze sentiment
    articles = news_data.get("articles", [])[:10]  # Limit to 10 articles
    analyzed_articles = []

    for article in articles:
        title = article.get("title", "No Title")
        sentiment_score = predict_sentiment(title)
        analyzed_articles.append({
            "title": title,
            "sentiment_score": sentiment_score,
            "link": article.get("url", "#")
        })

    return analyzed_articles

# ✅ API Endpoint to Get Average Sentiment
@app.get("/avg_sentiment/")
async def get_avg_sentiment(ticker: str):
    """
    Returns the average sentiment score for a stock based on headlines.
    """
    print(f"Fetching avg sentiment for ticker: {ticker}")

    # ✅ Fetch news and calculate average sentiment
    news_sentiment = await get_stock_news_sentiment(ticker)
    if "error" in news_sentiment:
        return news_sentiment  # Return error if news couldn't be fetched

    sentiments = [article["sentiment_score"] for article in news_sentiment]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

    return {
        "avg_sentiment": avg_sentiment,
        "date_range": {"start_date": "2025-02-12", "end_date": "2025-02-16"},
        "number_of_articles": len(sentiments),
    }

# ✅ Health Check Endpoint
@app.get("/")
async def health_check():
    return {"status": "API is running"}
