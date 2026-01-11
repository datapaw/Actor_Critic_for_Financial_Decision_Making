"""Add forward_return_X_1 and forward_return_X_2 columns (shifted versions) for all existing forward_return_X columns"""
import pandas as pd
from pathlib import Path
import re
import numpy as np
from sklearn.preprocessing import RobustScaler

def calculate_rsi(data, period=14):
	"""Calculate Relative Strength Index"""
	delta = data.diff()
	gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
	loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
	rs = gain / loss
	rsi = 100 - (100 / (1 + rs))
	return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
	"""Calculate MACD"""
	ema_fast = data.ewm(span=fast).mean()
	ema_slow = data.ewm(span=slow).mean()
	macd_line = ema_fast - ema_slow
	signal_line = macd_line.ewm(span=signal).mean()
	histogram = macd_line - signal_line
	return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, period=20, num_std=2):
	"""Calculate Bollinger Bands"""
	sma = data.rolling(window=period).mean()
	std = data.rolling(window=period).std()
	upper_band = sma + (std * num_std)
	lower_band = sma - (std * num_std)
	bb_width = (upper_band - lower_band) / sma
	bb_position = (data - lower_band) / (upper_band - lower_band)
	return upper_band, lower_band, bb_width, bb_position

def calculate_atr(high, low, close, period=14):
	"""Calculate Average True Range"""
	tr1 = high - low
	tr2 = abs(high - close.shift())
	tr3 = abs(low - close.shift())
	tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
	atr = tr.rolling(window=period).mean()
	return atr

def calculate_momentum(data, period=10):
	"""Calculate Momentum (Rate of Change)"""
	momentum = data.pct_change(periods=period) * 100
	return momentum

# Load prepared dataset
data_dir = Path('data/dataset')
prep_path = data_dir / 'sp500_prep.csv'

print(f"Loading data from {prep_path}...")
df = pd.read_csv(prep_path, index_col=0)

print(f"Current columns: {df.columns.tolist()}")
print(f"Current shape: {df.shape}")

# Find all existing forward_return_X columns and extract the days
print("\nFinding existing forward_return columns...")
existing_forward_cols = [col for col in df.columns if col.startswith('forward_return_') and '_' not in col.split('forward_return_')[1]]
existing_days = []

for col in existing_forward_cols:
    match = re.search(r'forward_return_(\d+)$', col)
    if match:
        days = int(match.group(1))
        existing_days.append(days)
        print(f"  Found: {col} (days={days})")

existing_days.sort()
print(f"\nExisting forward return periods: {existing_days}")

# Add shifted columns for X_1 and X_2 days for each existing X
print("\nCreating shifted forward return columns...")
for days in existing_days:
    col_original = f'forward_return_{days}'
    
    # Create X_1 (shifted by 1 day = next day's value)
    col_shift1 = f'forward_return_{days}_1'
    df[col_shift1] = df[col_original].shift(-1)
    print(f"  Created {col_shift1} (from next day)")
    
    # Create X_2 (shifted by 2 days = 2 days later value)
    col_shift2 = f'forward_return_{days}_2'
    df[col_shift2] = df[col_original].shift(-2)
    print(f"  Created {col_shift2} (from 2 days later)")

# ============================================================================
# ADD TECHNICAL INDICATORS
# ============================================================================
print("\n" + "="*80)
print("ADDING TECHNICAL INDICATORS")
print("="*80)

if 'Close' in df.columns:
    print("\nCalculating RSI (Relative Strength Index)...")
    df['rsi_14'] = calculate_rsi(df['Close'], period=14)
    print("  Created rsi_14")

if 'Close' in df.columns:
    print("\nCalculating MACD...")
    macd_line, signal_line, histogram = calculate_macd(df['Close'], fast=12, slow=26, signal=9)
    df['macd_line'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_histogram'] = histogram
    print("  Created macd_line, macd_signal, macd_histogram")

if 'Close' in df.columns:
    print("\nCalculating Bollinger Bands...")
    upper, lower, width, position = calculate_bollinger_bands(df['Close'], period=20, num_std=2)
    df['bb_upper'] = upper
    df['bb_lower'] = lower
    df['bb_width'] = width
    df['bb_position'] = position
    print("  Created bb_upper, bb_lower, bb_width, bb_position")

if 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
    print("\nCalculating ATR (Average True Range)...")
    df['atr_14'] = calculate_atr(df['High'], df['Low'], df['Close'], period=14)
    print("  Created atr_14")

if 'Close' in df.columns:
    print("\nCalculating Momentum (ROC)...")
    df['momentum_10'] = calculate_momentum(df['Close'], period=10)
    print("  Created momentum_10")

# Moving Averages
if 'Close' in df.columns:
    print("\nCalculating Moving Averages...")
    df['sma_5'] = df['Close'].rolling(window=5).mean()
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['ema_12'] = df['Close'].ewm(span=12).mean()
    print("  Created sma_5, sma_20, ema_12")

# Price Position (normalized between recent high/low)
if 'Close' in df.columns and 'High' in df.columns and 'Low' in df.columns:
    print("\nCalculating Price Position...")
    recent_high = df['High'].rolling(window=20).max()
    recent_low = df['Low'].rolling(window=20).min()
    df['price_position'] = (df['Close'] - recent_low) / (recent_high - recent_low)
    print("  Created price_position")

# Volatility (20-day returns std)
if 'Close' in df.columns:
    print("\nCalculating Volatility...")
    df['volatility_20'] = df['Close'].pct_change().rolling(window=20).std() * 100
    print("  Created volatility_20")

# Volume indicators
if 'Volume' in df.columns and 'Close' in df.columns:
    print("\nCalculating Volume indicators...")
    df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
    print("  Created volume_sma_20, volume_ratio")

print(f"\nUpdated columns: {df.columns.tolist()}")
print(f"Updated shape: {df.shape}")

# Add strategy class columns
print("\nCreating strategy class columns...")
for days in existing_days:
    col_0 = f'forward_return_{days}'          # Now (0 days wait)
    col_1 = f'forward_return_{days}_1'        # Next day (1 day wait)
    col_2 = f'forward_return_{days}_2'        # 2 days later (2 days wait)
    
    class_col = f'strategy_class_{days}'
    
    # Create dataframe with the three return columns for comparison
    comparison = df[[col_0, col_1, col_2]].copy()
    
    # Find argmax (0, 1, or 2) for each row
    df[class_col] = np.argmax(comparison.values, axis=1)
    
    # Count the strategy distribution
    unique, counts = np.unique(df[class_col], return_counts=True)
    print(f"\n  Created {class_col}:")
    for strategy, count in zip(unique, counts):
        pct = (count / len(df) * 100) if len(df) > 0 else 0
        wait_days = f"{strategy} day(s)" if strategy != 0 else "Now (0 days)"
        print(f"    Class {strategy} ({wait_days}): {count} samples ({pct:.2f}%)")

# Add strategy confidence and margin metrics
print("\nCreating strategy confidence and margin columns...")
for days in existing_days:
    col_0 = f'forward_return_{days}'          # Now (0 days wait)
    col_1 = f'forward_return_{days}_1'        # Next day (1 day wait)
    col_2 = f'forward_return_{days}_2'        # 2 days later (2 days wait)
    
    confidence_col = f'strategy_confidence_{days}'
    margin_col = f'strategy_margin_{days}'
    
    # Get returns for all three options
    returns = df[[col_0, col_1, col_2]].values
    
    # Find max and second max for each row
    max_returns = np.max(returns, axis=1)
    # To get second max, sort and take second largest value
    sorted_returns = np.sort(returns, axis=1)
    second_max_returns = sorted_returns[:, -2]
    
    # Confidence = difference between best and second best (how much better is the best choice)
    df[confidence_col] = max_returns - second_max_returns
    
    # Margin = normalized confidence (confidence / abs(best_return) to account for scale)
    # Avoid division by zero by adding small epsilon
    df[margin_col] = df[confidence_col] / (np.abs(max_returns) + 1e-10)
    
    print(f"  Created {confidence_col} (absolute margin)")
    print(f"  Created {margin_col} (normalized margin)")

print(f"\nFinal columns: {df.columns.tolist()}")
print(f"Final shape: {df.shape}")

# Display summary
print("\n" + "="*50)
print("Strategy Metrics Summary:")
print("="*50)
for days in existing_days:
    confidence_col = f'strategy_confidence_{days}'
    margin_col = f'strategy_margin_{days}'
    
    print(f"\n{days}-day period:")
    print(f"  Confidence (absolute margin):")
    print(f"    Mean: {df[confidence_col].mean():.6f}")
    print(f"    Std:  {df[confidence_col].std():.6f}")
    print(f"    Min:  {df[confidence_col].min():.6f}")
    print(f"    Max:  {df[confidence_col].max():.6f}")
    print(f"  Margin (normalized):")
    print(f"    Mean: {df[margin_col].mean():.6f}")
    print(f"    Std:  {df[margin_col].std():.6f}")
    print(f"    Min:  {df[margin_col].min():.6f}")
    print(f"    Max:  {df[margin_col].max():.6f}")

print("\n" + "="*50)
print("Forward Return Columns Summary:")
print("="*50)
forward_cols = [col for col in df.columns if col.startswith('forward_return_')]
forward_cols.sort(key=lambda x: (int(x.split('_')[2]), int(x.split('_')[3]) if len(x.split('_')) > 3 else 0))
for col in forward_cols:
    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.6f}")
    print(f"  Std:  {df[col].std():.6f}")
    print(f"  Min:  {df[col].min():.6f}")
    print(f"  Max:  {df[col].max():.6f}")
    print(f"  NaNs: {df[col].isna().sum()}")

# ============================================================================
# NORMALIZATION - RobustScaler (better for financial data with outliers)
# ============================================================================
print("\n" + "="*80)
print("NORMALIZING FEATURES (RobustScaler)")
print("="*80)

# Identify numeric columns (exclude Date and any string columns)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNormalizing {len(numeric_cols)} numeric columns with RobustScaler...")
print("Method: (x - median) / IQR")
print("Advantages: Resistant to outliers, preserves data distribution")

# Fill NaN values with column median first
for col in numeric_cols:
	if df[col].isna().any():
		df[col] = df[col].fillna(df[col].median())

# Apply RobustScaler (median-based, outlier-resistant)
scaler = RobustScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(f"\n✓ RobustScaler normalization complete!")
print(f"  Features centered around median with IQR scaling")

# Verify normalization statistics
print("\nNormalized Features Statistics:")
print(f"  Mean of all features:  {df[numeric_cols].mean().mean():.6f}")
print(f"  Median of all features: {df[numeric_cols].median().median():.6f}")
print(f"  Std of all features:   {df[numeric_cols].std().mean():.6f}")
print(f"  Min value:  {df[numeric_cols].min().min():.6f}")
print(f"  Max value:  {df[numeric_cols].max().max():.6f}")
print(f"  (Most values should be roughly in [-2, 2] range, outliers beyond)")

# ============================================================================
# SAVE NORMALIZED DATA
# ============================================================================
output_path = data_dir / 'sp500_critic.csv'
df.to_csv(output_path, index=False)
print(f"\n✓ Normalized dataset saved to {output_path}")

print("\n✓ Complete! Created sp500_critic.csv with:")
print("  - Shifted forward returns (X_1, X_2)")
print("  - Strategy classes (which wait period is best)")
print("  - Strategy confidence (absolute return margin)")
print("  - Strategy margin (normalized return margin)")
print("  - Technical Indicators:")
print("    • RSI (14-period relative strength)")
print("    • MACD (trend following)")
print("    • Bollinger Bands (volatility & support/resistance)")
print("    • ATR (volatility)")
print("    • Momentum (rate of change)")
print("    • Moving Averages (trend)")
print("    • Price Position (within recent range)")
print("    • Volatility (20-day std of returns)")
print("    • Volume indicators (volume strength)")
print("  - ALL NUMERIC FEATURES NORMALIZED (RobustScaler)")
print("    • Median-centered, IQR-scaled")
print("    • Resistant to outliers")

