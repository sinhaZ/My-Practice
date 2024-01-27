# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt



# Load stock price data 
data = pd.read_csv('stock_data.csv')
# Convert the date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date
data = data.sort_values('Date')

# Feature engineering: Create additional features if needed
# For example, calculate moving averages
data['MA5'] = data['Close'].rolling(window=5).mean()

# Create target variable (next day's closing price)
data['Target'] = data['Close'].shift(-1)

# Drop rows with missing values
data = data.dropna()

# Select features and target variable
features = ['Close', 'MA5']  # Adjust as needed
X = data[features]
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model (Random Forest Regressor in this example)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

# Visualize predicted vs. actual prices
plt.figure(figsize=(12, 6))
plt.plot(data['Date'][-len(predictions):], y_test.values, label='Actual Prices')
plt.plot(data['Date'][-len(predictions):], predictions, label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
