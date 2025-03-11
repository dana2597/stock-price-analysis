# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Step 1: Fetch Historical Stock Data
def fetch_stock_data(ticker_symbol, period="6mo", interval="1d"):
    print(f"Fetching stock data for {ticker_symbol}...")
    stock = yf.Ticker(ticker_symbol)
    data = stock.history(period=period, interval=interval)
    return data

# Step 2: Preprocess Data
def preprocess_data(data):
    data["Date"] = data.index
    data["Date"] = pd.to_datetime(data["Date"])
    data["Day"] = data["Date"].dt.day
    data["Month"] = data["Date"].dt.month
    data["Year"] = data["Date"].dt.year
    return data

# Step 3: Train Linear Regression Model
def train_model(data):
    # Select features (Day, Month, Year) and target (Close price)
    X = data[["Day", "Month", "Year"]]
    y = data["Close"]

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Model R-squared score: {score:.2f}")
    return model

# Step 4: Predict Future Prices
def predict_future_prices(model, days_ahead=30):
    # Generate future dates
    future_dates = [(datetime.now() + timedelta(days=i)).date() for i in range(1, days_ahead + 1)]
    future_df = pd.DataFrame(future_dates, columns=["Date"])
    future_df["Day"] = future_df["Date"].apply(lambda x: x.day)
    future_df["Month"] = future_df["Date"].apply(lambda x: x.month)
    future_df["Year"] = future_df["Date"].apply(lambda x: x.year)

    # Predict prices for future dates
    predicted_prices = model.predict(future_df[["Day", "Month", "Year"]])
    future_df["Predicted Price"] = predicted_prices
    return future_df

# Step 5: Visualize Results
def visualize_data(data, future_df):
    plt.figure(figsize=(10, 6))

    # Historical data
    plt.plot(data["Date"], data["Close"], label="Historical Prices", color="blue")

    # Predicted future prices
    plt.plot(future_df["Date"], future_df["Predicted Price"], label="Predicted Prices", color="orange", linestyle="--")

    plt.title("Stock Price Analysis and Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main Function
if __name__ == "__main__":
    ticker = "^NSEI"  # Replace with desired stock symbol (e.g., "^BSESN" for Sensex)
    data = fetch_stock_data(ticker)

    # Preprocess data
    processed_data = preprocess_data(data)
    print("Processed Data Preview:")
    print(processed_data.head())

    # Train the model
    model = train_model(processed_data)

    # Predict future prices
    future_predictions = predict_future_prices(model)
    print("\nFuture Predictions:")
    print(future_predictions)

    # Visualize the data
    visualize_data(processed_data, future_predictions)
