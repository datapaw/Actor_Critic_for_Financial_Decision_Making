import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/dataset/sp500_prep.csv')

print("="*80)
print("VERIFYING FORWARD_RETURN DATA FORMAT")
print("="*80)
print()
print("Testing: Is forward_return in POINTS or PERCENT?")
print()

# Sample rows where we can check 10 days ahead
sample_indices = [100, 200, 300, 400, 500]

print("Method 1: Check if Close + forward_return_10 = Close_after_10_days")
print("-"*80)
for idx in sample_indices:
    if idx + 10 >= len(df):
        continue
    
    row = df.iloc[idx]
    future_row = df.iloc[idx + 10]
    
    close_now = row['Close']
    forward_ret = row['forward_return_10']
    close_future = future_row['Close']
    
    # Test 1: If it's in POINTS, close + forward_return should match future close
    predicted_close_points = close_now + forward_ret
    error_points = abs(predicted_close_points - close_future)
    
    # Test 2: If it's in PERCENT, close * (1 + forward_return/100) should match
    predicted_close_pct = close_now * (1 + forward_ret / 100)
    error_pct = abs(predicted_close_pct - close_future)
    
    print(f"Row {idx}:")
    print(f"  Current Close: {close_now:.2f}")
    print(f"  forward_return_10: {forward_ret:.2f}")
    print(f"  Actual Close after 10 days: {close_future:.2f}")
    print(f"  ")
    print(f"  If POINTS: {close_now:.2f} + {forward_ret:.2f} = {predicted_close_points:.2f} (error: {error_points:.2f})")
    print(f"  If PERCENT: {close_now:.2f} * (1 + {forward_ret:.2f}/100) = {predicted_close_pct:.2f} (error: {error_pct:.2f})")
    print(f"  ✓ Best match: {'POINTS' if error_points < error_pct else 'PERCENT'}")
    print()

print()
print("="*80)
print("CHECKING YOUR EXAMPLE: Buy at 1500, sell at 1544")
print("="*80)
print()
print("If forward_return = 44.0 (the DIFFERENCE in price):")
print("  Return in POINTS: 44.0")
print("  Return in PERCENT: (44.0 / 1500) * 100 = 2.93%")
print()
print("My current code does:")
print("  return_pct = (forward_return_points / close_price) * 100")
print("  return_pct = (44.0 / 1500) * 100 = 2.93%")
print()
print("This assumes forward_return is in POINTS (absolute difference).")
print("Let's verify with actual data above...")
