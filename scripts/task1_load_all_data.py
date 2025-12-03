"""
Task 1 - Full Dataset: Load and Filter ALL FNSPID Dataset
Load all news data from HuggingFace (not just 2020) and perform initial filtering.
WARNING: This will process 13+ million rows and take significant time.
"""

import pandas as pd
from huggingface_hub import hf_hub_download
import time

print("=" * 60)
print("TASK 1: LOAD AND FILTER ALL FNSPID DATASET")
print("=" * 60)

start_time = time.time()

# Download the CSV file from HuggingFace
print("\nDownloading FNSPID dataset from HuggingFace...")
file_path = hf_hub_download(
    repo_id="Zihan1004/FNSPID",
    filename="Stock_news/All_external.csv",
    repo_type="dataset"
)

# Load with pandas, being flexible with data types
print("Loading CSV with pandas (this may take a few minutes)...")
df = pd.read_csv(file_path, low_memory=False)
print(f"Total rows loaded: {len(df):,}")

# Convert Date column to datetime
print("Parsing dates...")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Remove rows with invalid dates
df = df.dropna(subset=['Date'])
print(f"Rows after removing invalid dates: {len(df):,}")

# Keep only necessary columns
df = df[['Date', 'Article_title', 'Stock_symbol']].copy()

# Remove rows with null article titles
df = df.dropna(subset=['Article_title'])
print(f"Rows after removing null titles: {len(df):,}")

# Remove rows with null stock symbols
df = df.dropna(subset=['Stock_symbol'])
print(f"Rows after removing null symbols: {len(df):,}")

# Remove duplicates (same article for same stock on same date)
original_len = len(df)
df = df.drop_duplicates()
print(f"Removed {original_len - len(df):,} duplicate rows")
print(f"Rows after deduplication: {len(df):,}")

# Sort by date for easier processing
df = df.sort_values('Date')

# Save cleaned dataset
output_file = '../data/fnspid_all_cleaned.csv'
print(f"\nSaving cleaned dataset to {output_file}...")
df.to_csv(output_file, index=False)

elapsed = time.time() - start_time
print(f"\n  Task 1 Complete!")
print(f"  Saved to: {output_file}")
print(f"  Final row count: {len(df):,}")
print(f"  Unique stocks: {df['Stock_symbol'].nunique():,}")
print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"  Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
