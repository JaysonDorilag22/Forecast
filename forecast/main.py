from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from statsmodels.tsa.arima.model import ARIMA
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize FastAPI app
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

# Pydantic models
class PPIForecast(BaseModel):
    date: str
    value: float

class ProductForecast(BaseModel):
    product_name: str
    current_price: float
    forecasts: List[Dict[str, Any]]

# Utility function to prepare PPI DataFrame
def fetch_ppi_df():
    ppis = list(db['forecasts'].find())
    if not ppis:
        raise ValueError("No PPI data found in database.")

    df = pd.DataFrame(ppis)
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df['month'] = df['month'].map(month_map)
    df['Date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df = df.last('12M')
    return df

@app.get("/", response_model=Dict[str, Any])
def root():
    return {"message": "Welcome to the Forecast API", "endpoints": ["/forecast/ppi", "/forecast/products"]}

@app.get("/forecast/ppi", response_model=List[PPIForecast])
def forecast_ppi(months: int = 12):
    try:
        ppi_df = fetch_ppi_df()
        model = ARIMA(ppi_df['index'], order=(2, 1, 2))
        model_fit = model.fit()
        forecast_values = model_fit.forecast(steps=months)
        future_dates = pd.date_range(start=ppi_df.index[-1] + pd.DateOffset(months=1), periods=months, freq='MS')

        return [
            {"date": date.strftime("%Y-%m-%d"), "value": round(float(value), 3)}
            for date, value in zip(future_dates, forecast_values)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecast/products", response_model=List[ProductForecast])
def forecast_products(months: int = 12):
    try:
        ppi_df = fetch_ppi_df()
        model = ARIMA(ppi_df['index'], order=(2, 1, 2))
        model_fit = model.fit()
        forecast_values = model_fit.forecast(steps=months)
        future_dates = pd.date_range(start=ppi_df.index[-1] + pd.DateOffset(months=1), periods=months, freq='MS')

        base_ppi = ppi_df['index'].iloc[-1]

        products = list(db['products'].find())
        if not products:
            raise ValueError("No product data found in database.")
        
        results = []
        for product in products:
            base_price = product['price']
            product_name = product['name']
            adjusted_prices = base_price * (forecast_values / base_ppi)
            forecasts = [
                {"date": date.strftime("%Y-%m-%d"), "price": round(float(price), 3)}
                for date, price in zip(future_dates, adjusted_prices)
            ]

            results.append({
                "product_name": product_name,
                "current_price": float(base_price),
                "forecasts": forecasts
            })
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
