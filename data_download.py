import yfinance as yf
import os
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

# Download S&P 500 data (10 years, 1-day frequency)
print("Downloading 10 years of S&P 500 data from Yahoo Finance...")
sp500 = yf.download('^GSPC', period='25y', interval='1d', progress=True)

# Save to CSV
output_path = data_dir / 'sp500_data.csv'
sp500.to_csv(output_path)

print(f"Data saved to {output_path}")
print(f"\nData shape: {sp500.shape}")
print(f"\nFirst few rows:")
print(sp500.head())

# Delete rows 1 and 2 from the CSV
import pandas as pd
df = pd.read_csv(output_path, index_col=0)
df = df.drop(df.index[0:2])  # Drop rows at index 0 and 1 (1st and 2nd rows)
df.to_csv(output_path)
print(f"\nDeleted rows 1 and 2 from CSV")
print(f"Updated data shape: {df.shape}")

# Rename index to 'Date' if it doesn't have a name
df.index.name = 'Date'
df.to_csv(output_path)
print(f"Index renamed to 'Date'")