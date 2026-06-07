import yfinance as yf
import os
from pathlib import Path

# make sure we have a place to put the data
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

# grab 25 years of S&P 500 data - this might take a minute
print("Pulling S&P 500 data from Yahoo Finance... (this could take a bit)")
sp500 = yf.download('^GSPC', period='25y', interval='1d', progress=True)

output_path = data_dir / 'sp500_data.csv'
sp500.to_csv(output_path)

print(f"\nGot it! Saved to {output_path}")
print(f"Total rows: {sp500.shape[0]}, columns: {sp500.shape[1]}")
print(f"\nHere's what the first few rows look like:")
print(sp500.head())

# cleanup - need to drop the first two rows for some reason (weird yahoo finance thing)
import pandas as pd
df = pd.read_csv(output_path, index_col=0)
df = df.drop(df.index[0:2])  # dropping first 2 rows
df.to_csv(output_path)
print(f"\nCleaned up first 2 rows")
print(f"New shape: {df.shape}")

# make sure the index is labeled as Date
df.index.name = 'Date'
df.to_csv(output_path)
print(f"Done! Index is now properly labeled as 'Date'")