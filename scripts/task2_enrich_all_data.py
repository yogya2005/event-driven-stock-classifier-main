"""
Task 2 - Full Dataset: Enrich with Stock Data from yfinance
Download stock metadata and price data for ALL stocks in the dataset.
Uses chunked processing and caching to handle large datasets efficiently.
WARNING: This may take several hours depending on the number of unique stocks.
"""

import pandas as pd
import yfinance as yf
import numpy as np
from datetime import timedelta
import time
import pickle
import os

print("=" * 60)
print("TASK 2: ENRICH ALL DATA WITH STOCK INFORMATION")
print("=" * 60)

start_time = time.time()

# Load cleaned dataset
print("\nLoading cleaned dataset...")
df = pd.read_csv('../data/fnspid_all_cleaned.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f"Loaded {len(df):,} rows")

# Get unique stocks
unique_stocks = df['Stock_symbol'].unique()
print(f"Unique stocks to process: {len(unique_stocks):,}")

# Check if we have cached data from previous runs
cache_file = 'stock_data_cache.pkl'
if os.path.exists(cache_file):
    print(f"\nFound cache file: {cache_file}")
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    stock_data = cache.get('stock_data', {})
    stock_metadata = cache.get('stock_metadata', {})
    failed_stocks = cache.get('failed_stocks', [])
    print(f"Loaded {len(stock_data)} cached stocks")
else:
    stock_data = {}
    stock_metadata = {}
    failed_stocks = []

# Determine date range needed
min_date = df['Date'].min()
max_date = df['Date'].max()
# Extend range to get next-day prices
fetch_start = min_date - timedelta(days=7)
fetch_end = max_date + timedelta(days=30)
print(f"\nDate range to fetch: {fetch_start.date()} to {fetch_end.date()}")

# Download stock data for stocks not yet in cache
stocks_to_fetch = [s for s in unique_stocks if s not in stock_data and s not in failed_stocks]
print(f"Stocks remaining to fetch: {len(stocks_to_fetch):,}")

if stocks_to_fetch:
    print("\nDownloading stock data (this will take time)...")
    for i, symbol in enumerate(stocks_to_fetch):
        if i % 100 == 0:
            print(f"  Processing stock {i:,}/{len(stocks_to_fetch):,}: {symbol} ({i/len(stocks_to_fetch)*100:.1f}%)")
            # Save cache every 100 stocks
            if i > 0:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'stock_data': stock_data,
                        'stock_metadata': stock_metadata,
                        'failed_stocks': failed_stocks
                    }, f)
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Download price data
            hist = ticker.history(start=fetch_start, end=fetch_end)
            if len(hist) > 0:
                stock_data[symbol] = hist
                
                # Get metadata
                info = ticker.info
                stock_metadata[symbol] = {
                    'sector': info.get('sector', 'Unknown'),
                    'market_cap': info.get('marketCap', np.nan),
                    'beta': info.get('beta', np.nan)
                }
            else:
                failed_stocks.append(symbol)
                
        except Exception as e:
            failed_stocks.append(symbol)
            continue
    
    # Final cache save
    print("\nSaving final cache...")
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'stock_data': stock_data,
            'stock_metadata': stock_metadata,
            'failed_stocks': failed_stocks
        }, f)

print(f"\nSuccessfully downloaded {len(stock_data):,} stocks")
print(f"Failed stocks: {len(failed_stocks):,}")

# Function to calculate next-day return
def get_next_day_return(symbol, date, stock_data):
    """
    Get the next trading day's return for a given stock and date.
    Returns None if data not available.
    """
    if symbol not in stock_data:
        return None
    
    hist = stock_data[symbol]
    
    try:
        # Find the date in the historical data
        if date not in hist.index:
            # Find the next available trading day
            future_dates = hist.index[hist.index >= date]
            if len(future_dates) == 0:
                return None
            date = future_dates[0]
        
        current_price = hist.loc[date, 'Close']
        
        # Get next trading day
        future_dates = hist.index[hist.index > date]
        if len(future_dates) == 0:
            return None
        
        next_date = future_dates[0]
        next_price = hist.loc[next_date, 'Close']
        
        # Calculate return
        return_pct = (next_price - current_price) / current_price
        return return_pct
        
    except Exception as e:
        return None

# Calculate returns for all rows (with progress updates)
print("\nCalculating next-day returns...")
chunk_size = 100000
returns_list = []

for chunk_start in range(0, len(df), chunk_size):
    chunk_end = min(chunk_start + chunk_size, len(df))
    print(f"  Processing rows {chunk_start:,} to {chunk_end:,} ({chunk_end/len(df)*100:.1f}%)")
    
    chunk = df.iloc[chunk_start:chunk_end]
    chunk_returns = chunk.apply(
        lambda row: get_next_day_return(row['Stock_symbol'], row['Date'], stock_data),
        axis=1
    )
    returns_list.append(chunk_returns)

df['next_day_return'] = pd.concat(returns_list)

# Remove rows where we couldn't get returns
original_len = len(df)
df = df.dropna(subset=['next_day_return'])
print(f"Rows with valid returns: {len(df):,} (removed {original_len - len(df):,})")

# Add stock metadata to each row
print("\nAdding stock metadata...")
def get_stock_metadata(symbol, stock_metadata):
    if symbol in stock_metadata:
        return stock_metadata[symbol]
    return {'sector': 'Unknown', 'market_cap': np.nan, 'beta': np.nan}

df['sector'] = df['Stock_symbol'].apply(lambda x: get_stock_metadata(x, stock_metadata)['sector'])
df['market_cap'] = df['Stock_symbol'].apply(lambda x: get_stock_metadata(x, stock_metadata)['market_cap'])
df['beta'] = df['Stock_symbol'].apply(lambda x: get_stock_metadata(x, stock_metadata)['beta'])

# Remove rows with missing metadata
original_len = len(df)
df = df.dropna(subset=['sector', 'market_cap', 'beta'])
print(f"Rows after adding metadata: {len(df):,} (removed {original_len - len(df):,})")

# Create labels using ±1.5% threshold
print("\nCreating labels...")
def label_return(return_pct):
    if return_pct > 0.015:
        return 1  # Positive
    elif return_pct < -0.015:
        return -1  # Negative
    else:
        return 0  # Neutral

df['label'] = df['next_day_return'].apply(label_return)

# Check label distribution
print("\nLabel distribution:")
print(df['label'].value_counts().sort_index())
print("\nLabel distribution (%):")
print(df['label'].value_counts(normalize=True).sort_index() * 100)

# Save enriched dataset
output_file = '../data/stock_news_all_labeled.csv'
print(f"\nSaving labeled dataset to {output_file}...")
df.to_csv(output_file, index=False)

elapsed = time.time() - start_time
print(f"\n  Task 2 Complete!")
print(f"  Saved to: {output_file}")
print(f"  Final row count: {len(df):,}")
print(f"  Unique stocks: {df['Stock_symbol'].nunique():,}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes, {elapsed/3600:.1f} hours)")
