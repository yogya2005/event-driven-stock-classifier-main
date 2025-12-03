import time
import yfinance as yf

def download_stock_data(symbol):
    try:
        # Add a small delay to respect API rate limits
        time.sleep(0.5)  # Sleep for 0.5 seconds between requests
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        return hist
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return None

# Example usage in a loop
for symbol in stock_symbols:
    data = download_stock_data(symbol)
    if data is not None:
        process_data(data)
