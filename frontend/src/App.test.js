import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "./App";
import axios from "axios";

jest.mock("axios");

test("renders Stock Sentiment Analysis heading", () => {
  render(<App />);
  expect(screen.getByText(/Stock Sentiment Analysis/i)).toBeInTheDocument();
});

test("allows user to input a stock ticker", () => {
  render(<App />);
  const input = screen.getByPlaceholderText(/Enter Stock Ticker/i);
  fireEvent.change(input, { target: { value: "AAPL" } });
  expect(input.value).toBe("AAPL");
});

test("fetches and displays sentiment data on button click", async () => {
  axios.post.mockResolvedValue({
    data: {
      sentiment_score: 0.7,
      sentiments: [0.8, -0.2, 0.1],
    },
  });

  render(<App />);
  const input = screen.getByPlaceholderText(/Enter Stock Ticker/i);
  fireEvent.change(input, { target: { value: "AAPL" } });

  const button = screen.getByText(/Analyze/i);
  fireEvent.click(button);

  await waitFor(() => expect(screen.getByText(/Overall Sentiment Score:/i)).toBeInTheDocument());

  expect(screen.getByText(/0.7000/)).toBeInTheDocument();
});

test("handles API failure gracefully", async () => {
  axios.post.mockRejectedValue(new Error("Network Error"));

  render(<App />);
  const input = screen.getByPlaceholderText(/Enter Stock Ticker/i);
  fireEvent.change(input, { target: { value: "AAPL" } });

  const button = screen.getByText(/Analyze/i);
  fireEvent.click(button);

  await waitFor(() => expect(screen.getByText(/Failed to fetch sentiment/i)).toBeInTheDocument());
});
