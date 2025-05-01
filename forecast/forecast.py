import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import itertools

# MongoDB connection
client = MongoClient('mongodb+srv://von_user:admin12345@cluster0.vfpokij.mongodb.net/newdbforuser?retryWrites=true&w=majority&appName=AtlasApp')
db = client['newdbforuser']

# Fetch PPI data from MongoDB
ppi_collection = db['ppis']
ppis = ppi_collection.find()

# Convert PPI data to DataFrame
ppi_df = pd.DataFrame(list(ppis))

# Map month names to numeric values
month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
             'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
ppi_df['month'] = ppi_df['month'].map(month_map)

# Convert year and month into a datetime column
ppi_df['Date'] = pd.to_datetime(ppi_df[['year', 'month']].assign(day=1))
ppi_df.set_index('Date', inplace=True)

# Keep only the last 12 months of data
ppi_df = ppi_df.sort_index().last('12M')
print("Fetched last 12 months PPI values:")
print(ppi_df[['index']])

# ARIMA model to forecast PPI for the next 12 months
model = ARIMA(ppi_df['index'], order=(2, 1, 2))
model_fit = model.fit()
ppi_forecast = model_fit.forecast(steps=12)
print("Predicted PPI values for next 12 months:")
print(ppi_forecast)

# Generate future dates
future_dates = pd.date_range(start=ppi_df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')

# Plot PPI Forecast
plt.figure(figsize=(12, 6))
plt.plot(ppi_df.index, ppi_df['index'], label='Historical PPI', color='blue')
plt.plot(future_dates, ppi_forecast, linestyle='dashed', color='red')
plt.scatter(future_dates, ppi_forecast, color='red', marker='o')  # Add dotted points
plt.xlabel('Date')
plt.ylabel('PPI Index')
plt.title('PPI Forecast for Next 12 Months')
plt.legend()
plt.grid(True)
plt.show()

# Fetch Product data from MongoDB
product_collection = db['products']
products = product_collection.find()
product_df = pd.DataFrame(list(products))

# Linear Regression for product price forecasting
plt.figure(figsize=(12, 10))  # Increased height for better resolution

# Define a color cycle
colors = itertools.cycle(plt.cm.tab20.colors)

for _, product in product_df.iterrows():
    product_name = product['name']
    latest_price = product['price']
    
    print(f"Current price of {product_name}: {latest_price}")
    
    # Use the latest PPI and price data to scale the forecasted prices
    latest_ppi = ppi_df['index'].iloc[-1]
    forecasted_prices = latest_price * (ppi_forecast / latest_ppi)
    
    # Round forecasted prices to 3 decimal places
    forecasted_prices = np.round(forecasted_prices, 3)
    
    print(f"Forecasted prices for {product_name} over 12 months:")
    print(forecasted_prices)
    
    # Get the next color from the color cycle
    color = next(colors)
    
    # Plot product price forecast
    plt.plot(future_dates, forecasted_prices, linestyle='dashed', color=color, label=product_name)
    plt.scatter(future_dates, forecasted_prices, color=color, marker='o')  # Add dotted points matching line color
    plt.scatter(ppi_df.index[-1], latest_price, color=color, marker='o')

plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.title('Forecasted Product Prices for the Next 12 Months')
plt.legend(loc='upper left', bbox_to_anchor=(1,1)) 
plt.grid(True)
plt.tight_layout()
plt.show()