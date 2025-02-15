import React from "react";
import { Card, ListGroup, Container } from "react-bootstrap";

const stocks = [
  { name: "AAPL", sentiment: "positive" },
  { name: "TSLA", sentiment: "negative" }
];

const StockList = () => {
  return (
    <Container className="mt-4">
      <h2>Stock Sentiments</h2>
      {stocks.map((stock, index) => (
        <Card key={index} className="mb-3">
          <Card.Body>
            <Card.Title>{stock.name}</Card.Title>
            <ListGroup>
              <ListGroup.Item variant={stock.sentiment === "positive" ? "success" : "danger"}>
                {stock.sentiment}
              </ListGroup.Item>
            </ListGroup>
          </Card.Body>
        </Card>
      ))}
    </Container>
  );
};

export default StockList;
