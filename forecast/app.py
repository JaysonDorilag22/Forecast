from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pymongo import MongoClient
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Any
import json
from datetime import datetime

app = FastAPI(title="Forecast API", description="API for PPI and product price forecasting")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
client = MongoClient('mongodb+srv://von_user:admin12345@cluster0.vfpokij.mongodb.net/newdbforuser?retryWrites=true&w=majority&appName=AtlasApp')
db = client['newdbforuser']

# Data models
class PPIForecast(BaseModel):
    date: str
    value: float

class ProductForecast(BaseModel):
    product_name: str
    current_price: float
    forecasts: List[Dict[str, Any]]

@app.get("/", response_model=Dict[str, str])
def read_root():
    return {"message": "Welcome to the Forecast API", "endpoints": ["/forecast/ppi", "/forecast/products"]}

@app.get("/forecast/ppi", response_model=List[PPIForecast])
def get_ppi_forecast(months: int = 12):
    """
    Get PPI forecast for the specified number of months
    """
    try:
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
        
        # ARIMA model to forecast PPI
        model = ARIMA(ppi_df['index'], order=(2, 1, 2))
        model_fit = model.fit()
        ppi_forecast = model_fit.forecast(steps=months)
        
        # Generate future dates
        future_dates = pd.date_range(start=ppi_df.index[-1] + pd.DateOffset(months=1), periods=months, freq='MS')
        
        # Format the response
        result = []
        for date, value in zip(future_dates, ppi_forecast):
            result.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(float(value), 3)
            })
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecast/products", response_model=List[ProductForecast])
def get_product_forecasts(months: int = 12):
    """
    Get price forecasts for all products for the specified number of months
    """
    try:
        # Fetch PPI data and forecast
        ppi_collection = db['ppis']
        ppis = ppi_collection.find()
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
        
        # ARIMA model to forecast PPI
        model = ARIMA(ppi_df['index'], order=(2, 1, 2))
        model_fit = model.fit()
        ppi_forecast = model_fit.forecast(steps=months)
        
        # Generate future dates
        future_dates = pd.date_range(start=ppi_df.index[-1] + pd.DateOffset(months=1), periods=months, freq='MS')
        
        # Fetch Product data from MongoDB
        product_collection = db['products']
        products = product_collection.find()
        product_df = pd.DataFrame(list(products))
        
        results = []
        for _, product in product_df.iterrows():
            product_name = product['name']
            latest_price = product['price']
            
            # Use the latest PPI and price data to scale the forecasted prices
            latest_ppi = ppi_df['index'].iloc[-1]
            forecasted_prices = latest_price * (ppi_forecast / latest_ppi)
            forecasted_prices = np.round(forecasted_prices, 3)
            
            forecasts = []
            for date, price in zip(future_dates, forecasted_prices):
                forecasts.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "price": float(price)
                })
            
            results.append({
                "product_name": product_name,
                "current_price": float(latest_price),
                "forecasts": forecasts
            })
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)