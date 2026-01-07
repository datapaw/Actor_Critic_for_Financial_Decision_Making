import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Normalization option
# 0: No normalization
# 1: Min-Max normalization (scales to 0-1)
# 2: Z-score normalization (mean=0, std=1)
# 3: Robust scaling (uses median and IQR, resistant to outliers)
NORMALIZATION_TYPE = 0

# Forward return periods (days) to calculate
FORWARD_RETURN_DAYS = [10, 15, 30, 60]

# Time frame window size (number of days to look back)
TIME_FRAME_DAYS = 20

# Load the S&P 500 data
data_dir = Path('data')
data_path = data_dir / 'sp500_data.csv'

print(f"Loading data from {data_path}...")
df = pd.read_csv(data_path, index_col=0)

# Add forward return columns based on FORWARD_RETURN_DAYS
print("Calculating forward returns...")
for days in FORWARD_RETURN_DAYS:
    col_name = f'forward_return_{days}'
    df[col_name] = df['Close'].pct_change(periods=-days)
    print(f"  Calculated {col_name}")

# Add close difference column (today's close - tomorrow's close)
print("Calculating close difference (today - tomorrow)...")
df['close_diff_next_day'] = df['Close'] - df['Close'].shift(-1)
print("  Calculated close_diff_next_day")

print(f"\nData shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nColumn names:")
print(df.columns.tolist())

# Apply normalization based on NORMALIZATION_TYPE
print(f"\nApplying normalization (type: {NORMALIZATION_TYPE})...")
price_columns = ['Open', 'High', 'Low', 'Close']

if NORMALIZATION_TYPE == 0:
    print("No normalization applied")
    
elif NORMALIZATION_TYPE == 1:
    print("Applying Min-Max normalization (0-1 range)...")
    scaler = MinMaxScaler()
    df[price_columns] = scaler.fit_transform(df[price_columns])
    print("Price columns normalized to 0-1 range")
    
elif NORMALIZATION_TYPE == 2:
    print("Applying Z-score normalization (mean=0, std=1)...")
    scaler = StandardScaler()
    df[price_columns] = scaler.fit_transform(df[price_columns])
    print("Price columns standardized with mean=0 and std=1")
    
elif NORMALIZATION_TYPE == 3:
    print("Applying Robust scaling (using median and IQR)...")
    scaler = RobustScaler()
    df[price_columns] = scaler.fit_transform(df[price_columns])
    print("Price columns scaled using median and IQR (robust to outliers)")
    
else:
    print(f"Unknown normalization type: {NORMALIZATION_TYPE}")

# Save the prepared dataset
output_path = data_dir / 'dataset/sp500_prep.csv'
df.to_csv(output_path)
print(f"\nPrepared dataset saved to {output_path}")

# Create binary columns for best performing forward return period
print("\nCreating multi-class target column...")
forward_return_cols = [f'forward_return_{days}' for days in FORWARD_RETURN_DAYS]

# Create a single multi-class target: which period has the best return
df['target_period'] = -1  # -1 for invalid/ambiguous cases

# For each row, find the maximum return and assign the period
for idx in df.index:
    returns = df.loc[idx, forward_return_cols]
    if not returns.isna().all():  # Check if not all values are NaN
        max_col = returns.idxmax()  # Get column name with maximum value
        period = int(max_col.split('_')[2])  # Extract period
        # Map period to class label
        period_index = FORWARD_RETURN_DAYS.index(period)
        df.loc[idx, 'target_period'] = period_index

# Also create binary columns if needed
print("Creating binary columns for each period...")
for days in FORWARD_RETURN_DAYS:
    df[f'forward_binary_{days}'] = (df['target_period'] == FORWARD_RETURN_DAYS.index(days)).astype(int)

# Replace all -1 values with 0
print("Replacing invalid target_period values (-1) with 0...")
invalid_count = (df['target_period'] == -1).sum()
if invalid_count > 0:
    print(f"  Found {invalid_count} rows with target_period = -1, replacing with 0")
    df.loc[df['target_period'] == -1, 'target_period'] = 0
else:
    print("  No invalid values found")

print("Multi-class target and binary columns created")
print(f"\nFinal data shape: {df.shape}")
print(f"\nColumn names:")
print(df.columns.tolist())

# Save the final dataset with binary columns
df.to_csv(output_path)
print(f"\nFinal dataset saved to {output_path}")

# Show data distribution for binary columns
print("\n" + "="*50)
print("Data Distribution for Target Classes:")
print("="*50)

# Show multi-class distribution
print("\nMulti-class Target Distribution:")
for i, days in enumerate(FORWARD_RETURN_DAYS):
    count = (df['target_period'] == i).sum()
    pct = (count / len(df) * 100) if len(df) > 0 else 0
    print(f"  Period {days} days (Class {i}): {count} ({pct:.2f}%)")

# Show binary column distribution
print("\nBinary Column Distribution:")
for days in FORWARD_RETURN_DAYS:
    col = f'forward_binary_{days}'
    count_1 = (df[col] == 1).sum()
    count_0 = (df[col] == 0).sum()
    total = count_1 + count_0
    pct_1 = (count_1 / total * 100) if total > 0 else 0
    print(f"\n{col}:")
    print(f"  Count 1: {count_1} ({pct_1:.2f}%)")
    print(f"  Count 0: {count_0} ({100-pct_1:.2f}%)")

# Create time frame sequences for neural network input
print("\n" + "="*50)
print(f"Creating {TIME_FRAME_DAYS}-day time frame sequences...")
print("="*50)

import numpy as np

# Features to use in the time frame
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# Create sequences
sequences = []
targets = []
valid_dates = []

# Start from TIME_FRAME_DAYS to ensure we have enough history
for i in range(TIME_FRAME_DAYS, len(df)):
    # Get the time frame (20 days of history)
    sequence = df.iloc[i-TIME_FRAME_DAYS:i][feature_columns].values
    
    # Get the target for this prediction day
    target = df.iloc[i]['target_period']
    
    # Only include if target is valid (not -1)
    if target != -1 and not np.isnan(target):
        sequences.append(sequence)
        targets.append(target)
        valid_dates.append(df.index[i])

# Convert to numpy arrays
sequences = np.array(sequences)
targets = np.array(targets)

print(f"\nCreated {len(sequences)} time frame sequences")
print(f"Sequence shape: {sequences.shape}")
print(f"  - Samples: {sequences.shape[0]}")
print(f"  - Time steps: {sequences.shape[1]} days")
print(f"  - Features per time step: {sequences.shape[2]}")
print(f"Targets shape: {targets.shape}")

# Save sequences and targets
sequences_path = data_dir / 'dataset/sequences.npy'
targets_path = data_dir / 'dataset/targets.npy'
dates_path = data_dir / 'dataset/dates.npy'

np.save(sequences_path, sequences)
np.save(targets_path, targets)
np.save(dates_path, valid_dates)

print(f"\nSequences saved to {sequences_path}")
print(f"Targets saved to {targets_path}")
print(f"Dates saved to {dates_path}")
