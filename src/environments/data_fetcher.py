import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================
# I. DATA PREPARATION (15 YEARS)
# ==========================================

def get_data(tickers, period="15y"):
    # Download closing prices
    data = yf.download(tickers, period=period)['Close']
    
    # Convert to monthly format (Last business day of the month)
    prices_m = data.resample('BM').last().dropna()
    return prices_m

def calculate_monthly_features(prices_df):
    features_list = []
    for asset in prices_df.columns:
        asset_prices = prices_df[asset]
        df = pd.DataFrame(index=asset_prices.index)
        
        # Monthly log returns
        df[f'{asset}_Ret'] = np.log(asset_prices / asset_prices.shift(1))
        
        # Monthly RSI (6-month window for reactivity)
        delta = asset_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
        rs = gain / loss
        df[f'{asset}_RSI'] = 100 - (100 / (1 + rs))
        
        # Trend: 3-month vs 12-month average
        df[f'{asset}_Trend'] = asset_prices.rolling(window=3).mean() - asset_prices.rolling(window=12).mean()
        
        features_list.append(df)
    
    full_features = pd.concat(features_list, axis=1).dropna()
    aligned_prices = prices_df.loc[full_features.index]
    return aligned_prices, full_features