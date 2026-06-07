# Merge the enriched features with price data
# combines sp500_tools_enriched and sp500_prep into one master file

import os
from pathlib import Path
import pandas as pd


# features we actually want to use for training
# TODO: might want to experiment with different feature combos
DEFAULT_FEATURES = [
	'feature_rsi_14','feature_rsi_7','feature_macd','feature_macd_signal','feature_macd_histogram',
	'feature_stoch_k','feature_stoch_d','feature_roc_12','feature_roc_5','feature_ema_5','feature_ema_12',
	'feature_ema_20','feature_atr_14','feature_bb_position','feature_bb_width','feature_volume_sma',
	'feature_volume_ratio','feature_trend_strength','close','volume','close_diff_next_day','feature_ema_26',
	'feature_ema_50','feature_sma_20','feature_volatility_20','feature_resistance','feature_support',
	'feature_midpoint','feature_cmf','feature_price_return_5d','feature_price_return_10d','feature_price_return_20d',
	'feature_ema_12_26_diff','feature_ema_26_50_diff','feature_distance_to_resistance','feature_distance_to_support',
	'feature_obv','feature_sma_50','feature_uptrend','feature_strong_uptrend','feature_resistance_60','feature_support_60',
	'feature_volatility_60','feature_autocorr_60','feature_return_60d','feature_return_30d','feature_volume_avg_60',
	'feature_sma_200','feature_sma_100','feature_price_sma20_diff','feature_price_sma50_diff','feature_resistance_30',
	'feature_support_30','feature_volume_trend','feature_hurst_exponent','avg_sentiment_val','avg_positive_score','avg_negative_score','avg_neutral_score'
]


def _detect_date_col(df: pd.DataFrame):
	# figure out which column is the date
	for cand in ('Date', 'date', 'date_only', 'datetime'):
		if cand in df.columns:
			return cand
	# try case-insensitive search
	for c in df.columns:
		if c.lower() in ('date', 'date_only', 'datetime'):
			return c
	return None


def load_csv_with_date(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	date_col = _detect_date_col(df)
	if date_col is None:
		raise ValueError(f"Can't find date column in {path}")
	df[date_col] = pd.to_datetime(df[date_col])
	# normalize to date (no time) for safe merging
	df['Date'] = df[date_col].dt.date
	return df


def merge_features_and_prices(enriched_path: Path, prep_path: Path, out_path: Path):
	enriched = load_csv_with_date(enriched_path)
	prep = load_csv_with_date(prep_path)

	# Prepare feature selection
	features_present = [f for f in DEFAULT_FEATURES if f in enriched.columns]
	missing_features = [f for f in DEFAULT_FEATURES if f not in enriched.columns]
	if missing_features:
		print(f"Warning: {len(missing_features)} DEFAULT_FEATURES not found in {enriched_path}: {missing_features[:6]}{'...' if len(missing_features)>6 else ''}")

	# Price columns to extract from prep
	price_cols = [c for c in ('High', 'Low', 'Open', 'Volume', 'target_period') if c in prep.columns]
	if not price_cols:
		raise ValueError(f"No price columns found in {prep_path}")

	# Select and deduplicate
	left = enriched[['Date'] + features_present].copy()
	right = prep[['Date'] + price_cols].copy()

	# Merge on Date (inner join to get aligned dates present in both)
	merged = pd.merge(left, right, on='Date', how='inner')

	# Sort by date ascending
	merged = merged.sort_values('Date')

	# Replace empty-string-only cells with NA, then fill NA with 0 for all columns except Date
	non_date_cols = [c for c in merged.columns if c != 'Date']
	merged[non_date_cols] = merged[non_date_cols].replace(r'^\s*$', pd.NA, regex=True)
	merged[non_date_cols] = merged[non_date_cols].fillna(0)

	# Apply StandardScaler normalization to all columns except Date and target_period
	from sklearn.preprocessing import StandardScaler
	cols_to_scale = [c for c in merged.columns if c not in ('Date', 'target_period')]
	if cols_to_scale:
		print(f"Applying StandardScaler to {len(cols_to_scale)} feature columns...")
		scaler = StandardScaler()
		merged[cols_to_scale] = scaler.fit_transform(merged[cols_to_scale])
		print("  Features normalized (mean=0, std=1)")

	out_path.parent.mkdir(parents=True, exist_ok=True)
	merged.to_csv(out_path, index=False)

	print(f"Saved merged file to: {out_path} (rows={len(merged)}, features={len(features_present)}, price_cols={price_cols})")


def main():
	repo_root = Path(__file__).resolve().parent
	enriched_path = repo_root / 'data' / 'dataset' / 'sp500_tools_enriched.csv'
	prep_path = repo_root / 'data' / 'dataset' / 'sp500_prep.csv'
	out_path = repo_root / 'data' / 'dataset' / 'sp500_features_prices_merged.csv'

	if not enriched_path.exists():
		raise FileNotFoundError(f"Expected enriched features file at {enriched_path}")
	if not prep_path.exists():
		raise FileNotFoundError(f"Expected prep prices file at {prep_path}")

	merge_features_and_prices(enriched_path, prep_path, out_path)


if __name__ == '__main__':
	main()

