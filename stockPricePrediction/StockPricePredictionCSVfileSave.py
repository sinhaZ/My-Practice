!pip install yfinance
import yfinance as yf
import pandas as pd

# Define the stock symbol and the time period
symbol = 'AAPL'  # Replace with the symbol of the stock you're interested in
start_date = '2020-01-01'
end_date = '2024-01-01'

# Download historical stock data
stock_data = yf.download(symbol, start=start_date, end=end_date)

# Display the downloaded data
print(stock_data.head())

# Save the data to a CSV file
stock_data.to_csv('stock_data.csv', index=True)
