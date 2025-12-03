# Event-Driven Stock Prediction: A Technical Guide

## 1. Introduction

This guide details the technical implementation of an event-driven stock classifier. The project aims to predict whether a stock's price will move up, down, or stay neutral based on news headlines. We utilized the FNSPID dataset and enriched it with historical stock data from Yahoo Finance to train three machine learning models: Logistic Regression, Random Forest, and XGBoost.

This document covers our data processing pipeline, feature engineering techniques, model training strategies (including handling class imbalance), and evaluation results.

## 2. Data Processing

### 2.1 Dataset and Cleaning
We started with the **FNSPID (Financial News and Stock Price Integration Dataset)**. The raw data contained news headlines and tickers. We filtered for the year 2020 to focus on a volatile market period.

**Key Steps:**
1.  **Loading**: Read the raw CSV file.
2.  **Cleaning**: Removed rows with missing headlines or invalid tickers.
3.  **Filtering**: Selected a subset of 20,000 rows to ensure manageable processing times.

### 2.2 Stock Data Enrichment & Batch Processing
To label our data, we needed the next-day stock returns. We used the `yfinance` API to download historical price data.

**Insight: Batch Processing**
Downloading data for thousands of stocks can consume significant memory. We implemented a batch processing approach, handling data in chunks of 100,000 rows. This ensured our pipeline remained stable and didn't crash due to memory overflows.

**Code Snippet: Batch Processing**
```python
# From guide_snippets/batch_processing.py
import pandas as pd

# Calculate returns for all rows (with progress updates)
print("\nCalculating next-day returns...")
chunk_size = 100000
returns_list = []

# Process the dataframe in chunks to avoid memory issues
for chunk_start in range(0, len(df), chunk_size):
    chunk_end = min(chunk_start + chunk_size, len(df))
    print(f"  Processing rows {chunk_start:,} to {chunk_end:,} ({chunk_end/len(df)*100:.1f}%)")
    
    chunk = df.iloc[chunk_start:chunk_end]
    chunk_returns = chunk.apply(
        lambda row: get_next_day_return(row['Stock_symbol'], row['Date'], stock_data),
        axis=1
    )
    returns_list.append(chunk_returns)

# Combine results from all chunks
df['next_day_return'] = pd.concat(returns_list)
```

### 2.3 Handling API Rate Limits
We faced strict rate limits from the Yahoo Finance API. To prevent our IP from being blocked, we introduced a timed delay between requests.

**Insight: API Rate Limiting**
By adding a small `time.sleep()` call between requests, we ensured consistent data retrieval without hitting 429 Too Many Requests errors.

**Code Snippet: API Rate Limiting**
```python
# From guide_snippets/api_rate_limiting.py
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
```

## 3. Feature Engineering

We transformed our raw text and metadata into numerical features for our models:

1.  **Text Features**: We used `TfidfVectorizer` to convert news headlines into 100 numerical features, capturing the importance of words like "gain", "loss", "surge", etc.
2.  **Sector Features**: We one-hot encoded the stock sectors (e.g., Technology, Healthcare) to allow the model to learn sector-specific trends.
3.  **Continuous Features**: We normalized Market Cap and Beta values to ensure they were on the same scale as other features.

## 4. Model Training

We trained three models to compare performance:
1.  **Logistic Regression**: A strong baseline for text classification.
2.  **Random Forest**: An ensemble method good at capturing non-linear relationships.
3.  **XGBoost**: A gradient boosting framework known for high performance.

### 4.1 Handling Class Imbalance
Our dataset had an uneven distribution of Positive, Neutral, and Negative labels. To prevent the models from being biased towards the majority class, we used class weighting.

**Insight: Class Weighting**
We set `class_weight='balanced'` in scikit-learn models and calculated sample weights for XGBoost. This penalizes the model more for misclassifying minority classes.

**Code Snippet: Class Imbalance Handling**
```python
# From guide_snippets/class_imbalance.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# 1. Logistic Regression with class_weight='balanced'
lr_model = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    class_weight='balanced'  # Automatically adjust weights inversely proportional to class frequencies
)

# 2. Random Forest with class_weight='balanced'
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced'  # Handle imbalance in tree construction
)

# 3. XGBoost with sample weights
# Calculate weights for each sample in the training set
sample_weights = compute_sample_weight('balanced', y_train)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    objective='multi:softmax',
    num_class=3
)
# Pass sample_weights during training
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
```

## 5. Evaluation and Results

We evaluated our models using Accuracy, Precision, Recall, and F1-Score.

**Model Comparison**
[Insert model_comparison.png here]

**Confusion Matrices**
[Insert confusion_matrices.png here]

**Key Findings:**
- **Random Forest** achieved the highest accuracy (~43.8%), significantly outperforming the random baseline of 33%.
- **Neutral** predictions were the most accurate, suggesting that news often doesn't lead to significant immediate price changes.
- **Class weighting** successfully prevented the models from ignoring the minority classes.

## 6. Troubleshooting

**Common Error 1: `yfinance` Download Failures**
- **Issue**: Scripts hang or fail when downloading stock data.
- **Fix**: Check your internet connection and ensure you aren't being rate-limited. Use the `api_rate_limiting.py` strategy.

**Common Error 2: Memory Errors**
- **Issue**: `MemoryError` during feature engineering or training.
- **Fix**: Reduce the `chunk_size` in the batch processing step or sample a smaller subset of the data.

## 7. Conclusion

This project demonstrated that news headlines contain predictive signal for short-term stock movements. By combining text analysis with financial metadata and robust engineering practices (batch processing, rate limiting, class balancing), we built a system that outperforms random chance. Future work could involve using more advanced NLP models like BERT or FinBERT for better text understanding.
