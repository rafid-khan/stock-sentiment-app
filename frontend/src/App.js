import React, { useState } from "react";
import axios from "axios";
import { Container, Button, Form, ListGroup, Spinner, Alert, Row, Col } from "react-bootstrap";

function App() {
  const [ticker, setTicker] = useState("");
  const [articles, setArticles] = useState([]);
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
      setArticles([]);

      console.log("Fetching sentiment for:", ticker);

      // Fetch news articles and sentiment scores from FastAPI
      const response = await axios.get(
        `https://stock-sentiment-api.up.railway.app/stock_news_sentiment/?ticker=${ticker}`
      );

      console.log("API Response:", response.data);

      if (response.data.length === 0) {
        setError("No sentiment data available for this ticker.");
        return;
      }

      setArticles(response.data);
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
          placeholder="Enter Stock Ticker (e.g., AAPL)"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
        />
        <Button variant="primary" onClick={fetchSentiment} className="ms-2">
          {loading ? <Spinner animation="border" size="sm" /> : "Analyze"}
        </Button>
      </Form>

      {/* Error Message */}
      {error && <Alert variant="danger" className="mt-3">{error}</Alert>}

      {/* Display Articles with Sentiments */}
      {articles.length > 0 && (
        <>
          {/* Overall Sentiment Score (Average of all articles) */}
          <div className="mt-4 text-center">
            <h3>
              Average Sentiment Score:{" "}
              <strong>
                {(
                  articles.reduce((sum, article) => sum + article.compound, 0) /
                  articles.length
                ).toFixed(4)}
              </strong>
            </h3>
            <p className="lead">
              {interpretSentiment(
                articles.reduce((sum, article) => sum + article.compound, 0) /
                  articles.length
              )}
            </p>
          </div>

          <ListGroup className="mt-4">
            <h4>Recent Headlines with Sentiments:</h4>
            {articles.map((article, index) => (
              <ListGroup.Item key={index}>
                <Row>
                  <Col md={8}>
                    <a
                      href={article.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="fw-bold"
                    >
                      {article.title}
                    </a>
                  </Col>
                  <Col md={4} className="text-end">
                    <span className={article.compound > 0 ? "text-success" : "text-danger"}>
                      Sentiment: {article.compound.toFixed(4)}
                    </span>
                  </Col>
                </Row>
              </ListGroup.Item>
            ))}
          </ListGroup>
        </>
      )}
    </Container>
  );
}

export default App;
