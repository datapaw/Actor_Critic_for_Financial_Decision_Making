import pandas as pd

df = pd.read_csv('data/dataset/sp500_prep.csv')

print("="*80)
print("FINDING WHAT forward_return_10 ACTUALLY MEANS")
print("="*80)

# Look at row 100
row_idx = 100
current_row = df.iloc[row_idx]

print(f"\nStarting row {row_idx}:")
print(f"  Date: {current_row['Date']}")
print(f"  Close: {current_row['Close']:.2f}")
print(f"  forward_return_10: {current_row['forward_return_10']:.2f}")

# Check multiple future rows to find which one matches
print(f"\nChecking future rows:")
for offset in range(1, 20):
    future_idx = row_idx + offset
    if future_idx >= len(df):
        break
    
    future_row = df.iloc[future_idx]
    price_diff = future_row['Close'] - current_row['Close']
    pct_return = (price_diff / current_row['Close']) * 100
    
    # Check if forward_return_10 matches this offset
    matches_diff = abs(current_row['forward_return_10'] - price_diff) < 1.0
    matches_pct = abs(current_row['forward_return_10'] - pct_return) < 0.1
    
    marker = ""
    if matches_diff:
        marker = " <- MATCHES PRICE DIFF!"
    elif matches_pct:
        marker = " <- MATCHES PERCENTAGE!"
    
    print(f"  +{offset:2d} days: Close={future_row['Close']:7.2f}, "
          f"Diff={price_diff:7.2f}, Pct={pct_return:6.2f}%{marker}")

print("\n" + "="*80)
print("Let me also check the data generation code...")
print("="*80)
print("Maybe forward_return is calculated in a preprocessing script?")
print("Looking for data_preparation.py or similar...")
