import pandas as pd

df = pd.read_csv('data/dataset/sp500_prep.csv')

print("="*80)
print("CHECKING WHAT forward_return ACTUALLY REPRESENTS")
print("="*80)

# Look at row 100 and compare with 10 days later
row_idx = 100
row = df.iloc[row_idx]
future_row = df.iloc[row_idx + 10]

print(f"\nRow {row_idx}:")
print(f"  Date: {row['Date']}")
print(f"  Close: {row['Close']:.2f}")
print(f"  forward_return_10: {row['forward_return_10']:.2f}")

print(f"\nRow {row_idx + 10} (10 days later):")
print(f"  Date: {future_row['Date']}")
print(f"  Close: {future_row['Close']:.2f}")

print(f"\nManual calculations:")
price_diff = future_row['Close'] - row['Close']
print(f"  Price difference: {future_row['Close']:.2f} - {row['Close']:.2f} = {price_diff:.2f}")

pct_return = (price_diff / row['Close']) * 100
print(f"  Percentage return: ({price_diff:.2f} / {row['Close']:.2f}) * 100 = {pct_return:.2f}%")

print(f"\nWhat's stored in forward_return_10: {row['forward_return_10']:.2f}")
print(f"\nDoes it match:")
print(f"  Price difference? {abs(row['forward_return_10'] - price_diff) < 0.01}")
print(f"  Percentage? {abs(row['forward_return_10'] - pct_return) < 0.01}")

print("\n" + "="*80)
print("YOUR EXAMPLE: Buy at 1500, sell at 1544")
print("="*80)
print(f"Price difference: 1544 - 1500 = 44")
print(f"Percentage return: (44 / 1500) * 100 = {(44/1500)*100:.2f}%")
print()
print("If forward_return stores the DIFFERENCE (44), then I need to convert to %")
print("If forward_return stores the PERCENTAGE (2.93%), then I use it directly")
