import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

print("📌 Connecting to MongoDB...")
client = MongoClient('mongodb+srv://von_user:admin12345@cluster0.vfpokij.mongodb.net/newdbforuser?retryWrites=true&w=majority&appName=AtlasApp')
db = client['newdbforuser']
print("✅ Connected to MongoDB")

# Fetch PPI data
print("📌 Fetching PPI data from 'forecasts' collection...")
ppi_collection = db['forecasts']
ppis = list(ppi_collection.find())
print(f"✅ Fetched {len(ppis)} PPI records")

ppi_df = pd.DataFrame(ppis)
month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
             'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
ppi_df['month'] = ppi_df['month'].map(month_map)
ppi_df['Date'] = pd.to_datetime(ppi_df[['year', 'month']].assign(day=1))
ppi_df.set_index('Date', inplace=True)
ppi_df.rename(columns={'index': 'PPI'}, inplace=True)
ppi_df = ppi_df.last('12M')

print("📌 Last 12 months PPI data:")
print(ppi_df[['PPI']])

# ARIMA forecast
print("📌 Building ARIMA model (2,1,2)...")
model = ARIMA(ppi_df['PPI'], order=(2, 1, 2))
model_fit = model.fit()
print("✅ ARIMA model fitted")

print("📌 Forecasting PPI for next 12 months...")
ppi_forecast = model_fit.forecast(steps=12)
print("✅ Forecast complete. Forecasted PPI values:")
print(ppi_forecast)

future_dates = pd.date_range(start=ppi_df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')

# Fetch products
print("📌 Fetching product data from 'products' collection...")
product_collection = db['products']
products = list(product_collection.find())
print(f"✅ Fetched {len(products)} products")

product_df = pd.DataFrame(products)
base_ppi = ppi_df['PPI'].iloc[-1]
print(f"ℹ Base PPI for price adjustment: {base_ppi:.2f}")

# Compute price projections
product_prices = {}
for _, row in product_df.iterrows():
    base_price = row['price']
    new_prices = base_price * (ppi_forecast.values / base_ppi)
    product_prices[row['name']] = new_prices
    print(f"✅ Computed projected prices for '{row['name']}':")
    for date, price in zip(future_dates, new_prices):
        print(f"   {date.strftime('%Y-%m')} -> PHP {price:.2f}")

# Plot
print("📌 Plotting PPI and price projections...")
plt.figure(figsize=(14, 7))
plt.plot(ppi_df.index, ppi_df['PPI'], label='Historical PPI', color='blue')
plt.plot(future_dates, ppi_forecast, linestyle='dashed', color='red', label='Forecasted PPI')

for name, prices in product_prices.items():
    plt.plot(future_dates, prices, marker='o', label=f"{name} Price")

plt.xlabel('Date')
plt.ylabel('Index / Price')
plt.title('PPI Forecast and Product Price Projections')
plt.legend()
plt.grid(True)
plt.show()
print("✅ Plot displayed")
