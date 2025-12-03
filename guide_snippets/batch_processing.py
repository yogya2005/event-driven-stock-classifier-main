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
