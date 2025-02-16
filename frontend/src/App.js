import React, { useState } from "react";
import axios from "axios";
import { Container, Button, Form, ListGroup, Spinner, Alert, Row, Col } from "react-bootstrap";

function App() {
  const [ticker, setTicker] = useState("");
  const [headlines, setHeadlines] = useState([]);
  const [sentiments, setSentiments] = useState([]);
  const [sentimentScore, setSentimentScore] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Function to interpret sentiment score
  const interpretSentiment = (score) => {
    if (score >= 0.5) return "😃 Positive sentiment overall. Investors are optimistic!";
    if (score >= 0.1) return "🙂 Slightly positive sentiment. Investors are hopeful.";
    if (score > -0.1) return "😐 Neutral sentiment. Market reaction is balanced.";
    if (score > -0.5) return "🙁 Slightly negative sentiment. Some concerns exist.";
    return "😡 Negative sentiment. Investors are worried!";
  };

  const fetchSentiment = async () => {
    try {
      setError(null);
      setLoading(true);

      console.log("Fetching headlines for:", ticker);

      // Fetch headlines from Yahoo Finance API
      const newsResponse = await axios.get(
        `https://yahoo-finance-api-data.p.rapidapi.com/news/list`,
        {
          params: { symbol: ticker, limit: "10" },
          headers: {
            "x-rapidapi-key": process.env.REACT_APP_RAPIDAPI_KEY,  
            "x-rapidapi-host": "yahoo-finance-api-data.p.rapidapi.com",
          },
        }
      );

      // Extract headlines
      const extractedHeadlines = newsResponse.data.data.main.stream.map(
        (article) => article.content.title
      );
      setHeadlines(extractedHeadlines);
      console.log("Headlines received:", extractedHeadlines);

      // Send headlines to FastAPI for sentiment analysis
      const sentimentResponse = await axios.post("http://127.0.0.1:8000/predict/", {
        ticker,
        headlines: extractedHeadlines,
      });

      console.log("Sentiment response:", sentimentResponse.data);

      // Update state with sentiment scores
      setSentimentScore(sentimentResponse.data.sentiment_score);
      setSentiments(sentimentResponse.data.sentiments);
    } catch (err) {
      setError("Failed to fetch sentiment. Try again later.");
      console.error("Error fetching sentiment:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container className="mt-5">
      <h1 className="text-primary text-center">Stock Sentiment Analysis</h1>

      {/* Stock Ticker Input */}
      <Form className="d-flex">
        <Form.Control
          type="text"
          placeholder="Enter Stock Ticker (e.g., META)"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
        />
        <Button variant="primary" onClick={fetchSentiment} className="ms-2">
          {loading ? <Spinner animation="border" size="sm" /> : "Analyze"}
        </Button>
      </Form>

      {/* Error Message */}
      {error && <Alert variant="danger" className="mt-3">{error}</Alert>}

      {/* Overall Sentiment Score */}
      {sentimentScore !== null && (
        <div className="mt-4 text-center">
          <h3>
            Overall Sentiment Score: <strong>{sentimentScore.toFixed(4)}</strong>
          </h3>
          <p className="lead">{interpretSentiment(sentimentScore)}</p>  {/* Show sentiment meaning */}
        </div>
      )}

      {/* Display Headlines with Sentiments */}
      {headlines.length > 0 && (
        <ListGroup className="mt-4">
          <h4>Recent Headlines with Sentiments:</h4>
          {headlines.map((headline, index) => (
            <ListGroup.Item key={index}>
              <Row>
                <Col md={8}>
                  <strong>{headline}</strong>
                </Col>
                <Col md={4} className="text-end">
                  <span className={sentiments[index] > 0 ? "text-success" : "text-danger"}>
                    {sentiments[index] !== undefined ? sentiments[index].toFixed(4) : "N/A"}
                  </span>
                </Col>
              </Row>
            </ListGroup.Item>
          ))}
        </ListGroup>
      )}
    </Container>
  );
}

export default App;