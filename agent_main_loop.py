"""Agent ensemble voter with optional tool-enrichment second pass.

Two-pass inference:
1. FIRST PASS: Run 6 models on original test data → voting + reasoning
2. CONFIDENCE CHECK:
   - If confident AND no tie → use first pass results (done)
   - If not confident OR tie present → enrich with tools, re-run models, new voting

Models loaded from `models/`:
- mlp_wide_best.keras, wide_deep_best.keras, resnet_mlp_best.keras
- mlp_small_best.keras, se_mlp_best.keras, stacked_gru_ln_best.keras

Data: sp500_features_prices_merged.csv + optional tool features from tools.py
Voting: majority vote with confidence-based tie-breaking.
Output: results/ensemble_voting_results.txt + ensemble_predictions.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# TensorFlow / Keras
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

try:
	import tensorflow as tf
	from tensorflow import keras
	print(f"TensorFlow {tf.__version__} loaded successfully")
except ImportError as e:
	print(f"Failed to import TensorFlow: {e}")
	tf = None
	keras = None
except Exception as e:
	print(f"Unexpected error importing TensorFlow: {type(e).__name__}: {e}")
	tf = None
	keras = None

# tools integration
try:
	from tools import create_default_features
except Exception:
	create_default_features = None


def _require_tf():
	if tf is None:
		raise ImportError('TensorFlow is required. Install with `pip install tensorflow`.')


def split_train_val_test(X, y, val_size=0.15, test_size=0.15, random_seed=42):
	"""Split data into train/val/test chronologically."""
	n = len(X)
	test_idx = int(n * (1 - test_size))
	val_idx = int(test_idx * (1 - val_size))
	
	X_train, y_train = X[:val_idx], y[:val_idx]
	X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
	X_test, y_test = X[test_idx:], y[test_idx:]
	
	return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_merged_dataset(path: str | Path = None):
	"""Replicates the loader from agent_main.py: returns X, y, input_cols, dates."""
	repo_root = Path(__file__).resolve().parent
	if path is None:
		path = repo_root / 'data' / 'dataset' / 'sp500_features_prices_merged.csv'
	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(f"Merged CSV not found at: {path}")

	df = pd.read_csv(path)

	date_col = None
	for cand in ('Date', 'date', 'date_only'):
		if cand in df.columns:
			date_col = cand
			break

	if date_col is not None:
		df[date_col] = pd.to_datetime(df[date_col])
		df['Date'] = df[date_col].dt.date

	if 'target_period' not in df.columns:
		raise ValueError('`target_period` column not found in merged CSV')

	input_cols = [c for c in df.columns if c not in ('Date', 'target_period')]

	X_df = df[input_cols].copy()
	X_df = X_df.replace(["", " ", None], np.nan).fillna(0)
	for col in X_df.columns:
		X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)

	y = pd.to_numeric(df['target_period'], errors='coerce')
	if y.isna().any():
		raise ValueError('Some target_period values could not be parsed as integers')

	X = X_df.values.astype(np.float32)
	y = y.astype(int).values
	dates = df['Date'] if 'Date' in df.columns else None
	return X, y, input_cols, dates


def load_best_models(models_dir: str | Path = None):
	"""Load the specified best models from the models directory."""
	_require_tf()
	repo_root = Path(__file__).resolve().parent
	models_dir = Path(models_dir) if models_dir else (repo_root / 'models')
	files = {
		'mlp_wide': models_dir / 'mlp_wide_best.keras',
		'wide_deep': models_dir / 'wide_deep_best.keras',
		'resnet_mlp': models_dir / 'resnet_mlp_best.keras',
		'mlp_small': models_dir / 'mlp_small_best.keras',
		'se_mlp': models_dir / 'se_mlp_best.keras',
		'stacked_gru_ln': models_dir / 'stacked_gru_ln_best.keras',
	}

	loaded = {}
	for name, path in files.items():
		if not path.exists():
			print(f"Warning: model file missing: {path}")
			continue
		try:
			loaded[name] = keras.models.load_model(path)
			print(f"Loaded model: {name} from {path}")
		except Exception as e:
			print(f"Error loading {name} from {path}: {e}")
	if not loaded:
		raise FileNotFoundError('No models could be loaded. Check files in models/.')
	return loaded


def ensemble_vote(all_predictions, all_probabilities):
	"""
	Run voting logic on batch predictions from all models.
	
	INPUT:
	  all_predictions: dict[model_name, np.array of shape (n_samples,)]
	  all_probabilities: dict[model_name, np.array of shape (n_samples, 4)]
	
	OUTPUT:
	  vote_results: dict with voting metrics
	"""
	n_samples = len(next(iter(all_predictions.values())))
	n_models = len(all_predictions)
	n_classes = 4
	
	ensemble_predictions = np.zeros(n_samples, dtype=int)
	vote_counts = np.zeros((n_samples, n_classes), dtype=int)
	ensemble_confidences = np.zeros(n_samples)
	vote_entropies = np.zeros(n_samples)
	majority_percentages = np.zeros(n_samples)
	aggregate_probs = np.zeros((n_samples, n_classes))
	per_model_confidences = {name: np.zeros(n_samples) for name in all_predictions.keys()}
	
	# Aggregate predictions and probabilities
	for model_name in all_predictions.keys():
		preds = all_predictions[model_name]
		probs = all_probabilities[model_name]
		
		for i in range(n_samples):
			vote_class = preds[i]
			vote_counts[i, vote_class] += 1
			per_model_confidences[model_name][i] = np.max(probs[i])
		
		aggregate_probs += probs
	
	aggregate_probs /= n_models
	
	for i in range(n_samples):
		votes = vote_counts[i]
		max_votes = np.max(votes)
		winning_classes = np.where(votes == max_votes)[0]
		
		if len(winning_classes) == 1:
			ensemble_predictions[i] = winning_classes[0]
		else:
			ensemble_predictions[i] = np.argmax(aggregate_probs[i])
		
		ensemble_confidences[i] = aggregate_probs[i, ensemble_predictions[i]]
		vote_entropies[i] = entropy(votes + 1e-10)
		majority_percentages[i] = (max_votes / n_models) * 100
	
	return {
		'ensemble_predictions': ensemble_predictions,
		'vote_counts': vote_counts,
		'ensemble_confidences': ensemble_confidences,
		'vote_entropies': vote_entropies,
		'majority_percentages': majority_percentages,
		'aggregate_probabilities': aggregate_probs,
		'per_model_confidences': per_model_confidences,
		'n_models': n_models,
		'n_samples': n_samples
	}


def calculate_confidence_metrics(vote_results):
	"""Calculate comprehensive confidence metrics from voting results."""
	metrics = {
		'mean_ensemble_confidence': np.mean(vote_results['ensemble_confidences']),
		'std_ensemble_confidence': np.std(vote_results['ensemble_confidences']),
		'min_ensemble_confidence': np.min(vote_results['ensemble_confidences']),
		'max_ensemble_confidence': np.max(vote_results['ensemble_confidences']),
		
		'mean_entropy': np.mean(vote_results['vote_entropies']),
		'std_entropy': np.std(vote_results['vote_entropies']),
		'mean_majority_percentage': np.mean(vote_results['majority_percentages']),
		'min_majority_percentage': np.min(vote_results['majority_percentages']),
		
		'prediction_distribution': np.bincount(vote_results['ensemble_predictions'], minlength=4),
		'high_agreement_count': np.sum(vote_results['vote_entropies'] < 0.5),
		'medium_agreement_count': np.sum((vote_results['vote_entropies'] >= 0.5) & (vote_results['vote_entropies'] < 1.0)),
		'low_agreement_count': np.sum(vote_results['vote_entropies'] >= 1.0),
	}
	
	return metrics


def compute_reasoning(vote_results, conf_thresh=0.6, entropy_thresh=1.0, majority_thresh=50.0):
	"""
	Compute reasoning for each sample based on tie/low-confidence conditions.
	Returns summary metrics and per-sample reasoning details.
	"""
	n = vote_results['n_samples']
	reasons = []
	tie_resolved_count = 0
	low_conf_count = 0
	margins = []
	
	for i in range(n):
		votes = vote_results['vote_counts'][i]
		max_votes = int(votes.max())
		tie = int((votes == max_votes).sum()) > 1
		probs = vote_results['aggregate_probabilities'][i]
		pred = int(vote_results['ensemble_predictions'][i])
		top_prob = float(probs[pred])
		# second-best class by probability
		second_class = int(np.argsort(probs)[-2])
		second_prob = float(probs[second_class])
		margin = float(top_prob - second_prob)
		margins.append(margin)
		
		low_conf = (top_prob < conf_thresh) or \
				   (vote_results['vote_entropies'][i] >= entropy_thresh) or \
				   (vote_results['majority_percentages'][i] < majority_thresh)
		
		if tie:
			tie_resolved_count += 1
		if low_conf:
			low_conf_count += 1
		
		if tie:
			method = 'tie_breaker'
			reason_text = f"Tie resolved via aggregate probability; margin {margin:.4f} vs class {second_class}."
		elif low_conf:
			method = 'low_confidence'
			reason_text = (
				f"Low confidence (p={top_prob:.4f}, entropy={vote_results['vote_entropies'][i]:.2f}, "
				f"majority={vote_results['majority_percentages'][i]:.1f}%)."
			)
		else:
			method = 'majority'
			reason_text = f"Clear majority; margin {margin:.4f} over class {second_class}."
		
		reasons.append({
			'index': i,
			'prediction': pred,
			'top_prob': top_prob,
			'second_class': second_class,
			'second_prob': second_prob,
			'margin': margin,
			'tie': tie,
			'low_conf': low_conf,
			'method': method,
			'reason_text': reason_text,
		})
	
	return {
		'tie_resolved_count': tie_resolved_count,
		'low_conf_count': low_conf_count,
		'mean_margin': float(np.mean(margins)) if margins else 0.0,
		'median_margin': float(np.median(margins)) if margins else 0.0,
		'per_sample': reasons,
	}


def _ensure_tools_results_generated():
	"""Run using_tools pipeline if tools CSV does not exist."""
	if TOOLS_OUTPUT_PATH.exists():
		return
	if ModelDrivenToolSelector is None:
		print("using_tools not available; cannot generate tool results.")
		return
	try:
		print("Generating tool results...")
		selector = ModelDrivenToolSelector(Path('models/resnet_sp500.keras'), Path('data/dataset/sp500_prep.csv'))
		results_df = selector.run_full_pipeline()
		selector.save_results(results_df, TOOLS_OUTPUT_PATH)
		print(f"Generated tools output at {TOOLS_OUTPUT_PATH}")
	except Exception as e:
		print(f"Warning: Failed to generate tools output: {e}")


def _map_horizon_to_class(h):
	"""Map tool horizon string to class id."""
	return {'60_days': 0, '30_days': 1, '15_days': 2, '10_days': 3}.get(h, -1)


def get_tool_votes_for_dates(date_list):
	"""Fetch tool-based horizon recommendations for given dates."""
	_ensure_tools_results_generated()
	if not TOOLS_OUTPUT_PATH.exists():
		return None, None
	try:
		tools_df = pd.read_csv(TOOLS_OUTPUT_PATH)
		if 'date' not in tools_df.columns:
			return None, None
		tools_df['date'] = pd.to_datetime(tools_df['date']).dt.normalize()
		date_to_row = {d: row for d, row in zip(tools_df['date'], tools_df.to_dict('records'))}
		classes, confidences = [], []
		for d in date_list:
			try:
				if isinstance(d, str):
					dd = pd.to_datetime(d).normalize()
				else:
					dd = pd.Timestamp(d).normalize()
			except Exception:
				dd = None
			if dd is None:
				classes.append(-1)
				confidences.append(0.0)
			else:
				row = date_to_row.get(dd)
				if row is None:
					classes.append(-1)
					confidences.append(0.0)
				else:
					cls = _map_horizon_to_class(row.get('time_horizon'))
					classes.append(cls if cls != -1 else -1)
					confidences.append(float(row.get('model_confidence', 0.7)))
		return np.array(classes, dtype=int), np.array(confidences, dtype=np.float32)
	except Exception as e:
		print(f"Warning: Could not read tools output: {e}")
		return None, None
	"""Run inference on the latest sample and print per-model predictions."""
	latest_X = X[-1:]
	print('\nInference on latest sample:')
	for name, model in models.items():
		probs = model.predict(latest_X, verbose=0)[0]
		pred = int(np.argmax(probs))
		print(f"  {name}: pred={pred}, probs={np.round(probs, 4)}")
	if y is not None:
		print(f"  True label: {int(y[-1])}")


def generate_tool_features(df_full, test_start_idx, test_size):
	"""
	Generate tool features from the full OHLCV dataframe.
	
	Args:
		df_full: Full DataFrame with OHLCV columns
		test_start_idx: Index where test set starts
		test_size: Number of test samples
	
	Returns:
		Normalized tool features array (n_test, n_tool_features) or None if generation fails
	"""
	if create_default_features is None:
		print("Warning: create_default_features not available")
		return None
	
	try:
		# Extract OHLCV columns
		ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
		if not all(col in df_full.columns for col in ohlcv_cols):
			print(f"Warning: Not all OHLCV columns found. Available: {df_full.columns.tolist()}")
			return None
		
		df_ohlcv = df_full[ohlcv_cols].copy()
		
		# Generate tool features (40+ indicators)
		print("  Computing technical indicators...")
		tool_features_all = create_default_features(df_ohlcv)
		
		# Extract test portion
		tool_features_test = tool_features_all.iloc[test_start_idx:test_start_idx + test_size].reset_index(drop=True)
		
		# Handle NaN and inf values
		tool_features_test = tool_features_test.replace([np.inf, -np.inf], np.nan).fillna(0)
		
		# Normalize with StandardScaler
		scaler = StandardScaler()
		tool_features_test_values = tool_features_test.values.astype(np.float32)
		tool_features_test_scaled = scaler.fit_transform(tool_features_test_values)
		
		print(f"  Tool features shape: {tool_features_test_scaled.shape}")
		return tool_features_test_scaled.astype(np.float32)
	
	except Exception as e:
		print(f"Error generating tool features: {e}")
		return None


def save_ensemble_to_critic(X, y, models, critic_csv_path=None):
	"""
	Run ensemble inference on ALL data and save predictions + strength metrics to sp500_critic.csv.
	
	Args:
		X: Full feature array (n_samples, n_features)
		y: Full target array (n_samples,)
		models: Dict of loaded models
		critic_csv_path: Path to sp500_critic.csv (will be updated with new columns)
	
	Returns:
		Updated DataFrame with ensemble voting columns
	"""
	repo_root = Path(__file__).resolve().parent
	if critic_csv_path is None:
		critic_csv_path = repo_root / 'data' / 'dataset' / 'sp500_critic.csv'
	
	critic_csv_path = Path(critic_csv_path)
	
	print("\n" + "="*80)
	print("SAVING ENSEMBLE PREDICTIONS TO CRITIC DATASET")
	print("="*80)
	
	# Run inference on ALL data
	print(f"\nRunning batch inference on all {len(X)} samples...")
	all_predictions = {}
	all_probabilities = {}
	
	for model_name, model in models.items():
		probs = model.predict(X, verbose=0)
		preds = np.argmax(probs, axis=1)
		all_predictions[model_name] = preds
		all_probabilities[model_name] = probs
		print(f"  {model_name}: completed")
	
	# Apply ensemble voting on all data
	print("\nApplying ensemble voting on all data...")
	vote_results = ensemble_vote(all_predictions, all_probabilities)
	
	n_samples = len(X)
	
	# Extract voting metrics
	ensemble_predictions = vote_results['ensemble_predictions']
	ensemble_confidences = vote_results['ensemble_confidences']
	vote_entropies = vote_results['vote_entropies']
	majority_percentages = vote_results['majority_percentages']
	aggregate_probs = vote_results['aggregate_probabilities']
	vote_counts = vote_results['vote_counts']
	
	# Compute prediction margins and normalized margins
	prediction_margins = []
	prediction_margins_normalized = []
	models_in_agreement_list = []
	agreement_percentages = []
	had_tie_list = []
	low_confidence_flags = []
	
	for i in range(n_samples):
		# Prediction margin: best_prob - second_best_prob
		probs = aggregate_probs[i]
		sorted_probs = np.sort(probs)
		top_prob = sorted_probs[-1]
		second_prob = sorted_probs[-2]
		margin = top_prob - second_prob
		prediction_margins.append(margin)
		
		# Normalized margin
		margin_norm = margin / (np.abs(top_prob) + 1e-10)
		prediction_margins_normalized.append(margin_norm)
		
		# Models in agreement (voting same as ensemble prediction)
		ensemble_pred = ensemble_predictions[i]
		agreement_count = sum(1 for m in all_predictions.values() if m[i] == ensemble_pred)
		models_in_agreement_list.append(agreement_count)
		agreement_percentages.append((agreement_count / len(models)) * 100)
		
		# Tie detection
		votes = vote_counts[i]
		max_votes = np.max(votes)
		tie = int((votes == max_votes).sum()) > 1
		had_tie_list.append(tie)
		
		# Low confidence flag
		low_conf = (ensemble_confidences[i] < 0.6) or \
		           (vote_entropies[i] >= 1.0) or \
		           (majority_percentages[i] < 50.0)
		low_confidence_flags.append(low_conf)
	
	# Load existing sp500_critic.csv or create new one
	if critic_csv_path.exists():
		df_critic = pd.read_csv(critic_csv_path)
		print(f"\nLoaded existing {critic_csv_path} with {len(df_critic)} rows")
		
		# Ensure same number of rows
		if len(df_critic) != n_samples:
			print(f"Warning: critic CSV has {len(df_critic)} rows but X has {n_samples} rows")
			# Pad or trim as needed
			if len(df_critic) < n_samples:
				df_critic = pd.concat([df_critic, pd.DataFrame(index=range(len(df_critic), n_samples))], ignore_index=True)
			else:
				df_critic = df_critic.iloc[:n_samples].reset_index(drop=True)
	else:
		print(f"Creating new critic dataset with {n_samples} rows")
		df_critic = pd.DataFrame(index=range(n_samples))
	
	# Add voting + strength columns
	df_critic['ensemble_prediction'] = ensemble_predictions
	df_critic['ensemble_confidence'] = ensemble_confidences
	df_critic['vote_entropy'] = vote_entropies
	df_critic['majority_percentage'] = majority_percentages
	df_critic['prediction_margin'] = prediction_margins
	df_critic['prediction_margin_normalized'] = prediction_margins_normalized
	df_critic['models_in_agreement'] = models_in_agreement_list
	df_critic['agreement_percentage'] = agreement_percentages
	df_critic['prob_class_0'] = aggregate_probs[:, 0]
	df_critic['prob_class_1'] = aggregate_probs[:, 1]
	df_critic['prob_class_2'] = aggregate_probs[:, 2]
	df_critic['prob_class_3'] = aggregate_probs[:, 3]
	df_critic['had_tie'] = had_tie_list
	df_critic['low_confidence_flag'] = low_confidence_flags
	
	# Save updated CSV
	df_critic.to_csv(critic_csv_path, index=False)
	print(f"\nSaved critic dataset to {critic_csv_path}")
	print(f"Added columns: ensemble_prediction, ensemble_confidence, vote_entropy, majority_percentage,")
	print(f"              prediction_margin, prediction_margin_normalized, models_in_agreement,")
	print(f"              agreement_percentage, prob_class_0-3, had_tie, low_confidence_flag")
	print(f"Total shape: {df_critic.shape}")
	
	return df_critic


def main():
	"""
	Two-pass inference with conditional tool enrichment:
	
	PASS 1 (Original Features):
	  - Run 6 models on original test features (67 dims)
	  - Apply ensemble voting with tie-breaking
	  - Compute confidence metrics + reasoning
	  
	CONFIDENCE CHECK:
	  - If (mean_confidence > 0.6 AND mean_entropy < 1.0 AND mean_majority > 50%) 
	    AND tie_resolved_count == 0:
	    → Use PASS 1 results; skip PASS 2
	  - Else:
	    → Proceed to PASS 2 for tool enrichment
	  
	PASS 2 (Tool-Enriched Features):
	  - Generate all tool features from OHLCV data
	  - Apply StandardScaler normalization
	  - Concatenate with original features (117 total dims)
	  - Re-run 6 models + ensemble voting
	  - Compare PASS 1 vs PASS 2 results
	
	CRITIC DATASET:
	  - After two-pass inference, save ensemble metrics to sp500_critic.csv for all data
	  - This dataset will be used to train a critic model
	"""
	X, y, cols, dates = load_merged_dataset()
	print('Loaded merged dataset:')
	print('  rows=', X.shape[0], 'cols=', X.shape[1])

	# Split data (chronological: 70% train, 12.75% val, 17.25% test)
	(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_train_val_test(X, y, random_seed=42)
	print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
	test_start_idx = len(X_train) + len(X_val)  # Index where test set starts in full X
	
	# Load models
	models = load_best_models()
	print(f"Total models loaded: {len(models)}\n")
	
	# ========================================================================
	# PASS 1: Inference on original features
	# ========================================================================
	print("="*80)
	print("PASS 1: Ensemble voting on original features")
	print("="*80)
	
	print(f"Running batch inference on {len(X_test)} test samples...")
	all_predictions_pass1 = {}
	all_probabilities_pass1 = {}
	
	for model_name, model in models.items():
		probs = model.predict(X_test, verbose=0)
		preds = np.argmax(probs, axis=1)
		all_predictions_pass1[model_name] = preds
		all_probabilities_pass1[model_name] = probs
		print(f"  {model_name}: completed")
	
	# Apply ensemble voting
	print("\nApplying ensemble voting...")
	vote_results_pass1 = ensemble_vote(all_predictions_pass1, all_probabilities_pass1)
	confidence_metrics_pass1 = calculate_confidence_metrics(vote_results_pass1)
	reasoning_pass1 = compute_reasoning(vote_results_pass1)
	
	print(f"\nPass 1 Summary:")
	print(f"  Mean Confidence: {confidence_metrics_pass1['mean_ensemble_confidence']:.4f}")
	print(f"  Mean Entropy: {confidence_metrics_pass1['mean_entropy']:.4f}")
	print(f"  Mean Majority %: {confidence_metrics_pass1['mean_majority_percentage']:.1f}%")
	print(f"  Tie-resolved samples: {reasoning_pass1['tie_resolved_count']}")
	print(f"  Low-confidence samples: {reasoning_pass1['low_conf_count']}")
	
	# ========================================================================
	# CONFIDENCE CHECK: Should we run PASS 2?
	# ========================================================================
	mean_conf = confidence_metrics_pass1['mean_ensemble_confidence']
	mean_entropy = confidence_metrics_pass1['mean_entropy']
	mean_majority = confidence_metrics_pass1['mean_majority_percentage']
	tie_count = reasoning_pass1['tie_resolved_count']
	
	is_confident = (mean_conf > 0.6) and (mean_entropy < 1.0) and (mean_majority > 50.0)
	has_ties = (tie_count > 0)
	
	should_use_pass2 = (not is_confident) or has_ties
	
	print("\n" + "="*80)
	print("CONFIDENCE CHECK:")
	print("="*80)
	print(f"  Confidence > 0.6? {mean_conf > 0.6} (actual: {mean_conf:.4f})")
	print(f"  Entropy < 1.0? {mean_entropy < 1.0} (actual: {mean_entropy:.4f})")
	print(f"  Majority > 50%? {mean_majority > 50.0} (actual: {mean_majority:.1f}%)")
	print(f"  No ties? {not has_ties} (ties: {tie_count})")
	print(f"\n  => Pass 1 is CONFIDENT: {is_confident}")
	print(f"  => Run Pass 2: {should_use_pass2}")
	
	if not should_use_pass2:
		print("\n✓ Pass 1 results are confident with no ties. Using Pass 1 results.")
		final_vote_results = vote_results_pass1
		final_confidence_metrics = confidence_metrics_pass1
		final_reasoning = reasoning_pass1
		final_all_predictions = all_predictions_pass1
		pass2_executed = False
	else:
		print("\n✗ Pass 1 results need refinement. Running Pass 2 with tool enrichment...")
		
		# ====================================================================
		# PASS 2: Inference on tool-enriched features
		# ====================================================================
		print("\n" + "="*80)
		print("PASS 2: Ensemble voting on tool-enriched features")
		print("="*80)
		
		# Load full dataset for tool feature generation
		repo_root = Path(__file__).resolve().parent
		df_path = repo_root / 'data' / 'dataset' / 'sp500_features_prices_merged.csv'
		df_full = pd.read_csv(df_path)
		
		# Generate tool features
		print("\nGenerating tool features...")
		tool_features = generate_tool_features(df_full, test_start_idx, len(X_test))
		
		if tool_features is None:
			print("Tool features generation failed. Falling back to Pass 1 results.")
			final_vote_results = vote_results_pass1
			final_confidence_metrics = confidence_metrics_pass1
			final_reasoning = reasoning_pass1
			final_all_predictions = all_predictions_pass1
			pass2_executed = False
		else:
			print(f"  Generated {tool_features.shape[1]} tool features")
			
			# Concatenate original + tool features
			X_test_enriched = np.hstack([X_test, tool_features])
			print(f"  Enriched test set shape: {X_test_enriched.shape}")
			
			# Run inference on enriched features
			print(f"\nRunning batch inference on enriched test set...")
			all_predictions_pass2 = {}
			all_probabilities_pass2 = {}
			
			for model_name, model in models.items():
				probs = model.predict(X_test_enriched, verbose=0)
				preds = np.argmax(probs, axis=1)
				all_predictions_pass2[model_name] = preds
				all_probabilities_pass2[model_name] = probs
				print(f"  {model_name}: completed")
			
			# Apply ensemble voting
			print("\nApplying ensemble voting on enriched features...")
			vote_results_pass2 = ensemble_vote(all_predictions_pass2, all_probabilities_pass2)
			confidence_metrics_pass2 = calculate_confidence_metrics(vote_results_pass2)
			reasoning_pass2 = compute_reasoning(vote_results_pass2)
			
			print(f"\nPass 2 Summary:")
			print(f"  Mean Confidence: {confidence_metrics_pass2['mean_ensemble_confidence']:.4f}")
			print(f"  Mean Entropy: {confidence_metrics_pass2['mean_entropy']:.4f}")
			print(f"  Mean Majority %: {confidence_metrics_pass2['mean_majority_percentage']:.1f}%")
			print(f"  Tie-resolved samples: {reasoning_pass2['tie_resolved_count']}")
			print(f"  Low-confidence samples: {reasoning_pass2['low_conf_count']}")
			
			# Use Pass 2 results
			final_vote_results = vote_results_pass2
			final_confidence_metrics = confidence_metrics_pass2
			final_reasoning = reasoning_pass2
			final_all_predictions = all_predictions_pass2
			pass2_executed = True
	
	# ========================================================================
	# Save results
	# ========================================================================
	results_dir = Path(__file__).resolve().parent / 'results'
	results_dir.mkdir(parents=True, exist_ok=True)
	
	print("\n" + "="*80)
	
	# ========================================================================
	# Compute classification metrics on final results
	# ========================================================================
	y_true = y_test
	y_pred = final_vote_results['ensemble_predictions']
	acc = accuracy_score(y_true, y_pred)
	prec_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
	prec_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
	cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
	report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3])
	
	# Save comprehensive results to txt
	with open(results_dir / 'ensemble_voting_results.txt', 'w') as f:
		f.write("="*80 + "\n")
		f.write("ENSEMBLE VOTING RESULTS - TWO-PASS INFERENCE\n")
		f.write("="*80 + "\n\n")
		
		# Pass 1 summary
		f.write("PASS 1 RESULTS (Original Features)\n")
		f.write("-"*80 + "\n")
		f.write(f"Test Set Size: {len(X_test)}\n")
		f.write(f"Number of Models: {vote_results_pass1['n_models']}\n")
		f.write(f"Input Dimensions: {X_test.shape[1]}\n\n")
		
		f.write(f"Pass 1 Confidence Metrics:\n")
		f.write(f"  Mean Confidence: {confidence_metrics_pass1['mean_ensemble_confidence']:.4f}\n")
		f.write(f"  Std Dev Confidence: {confidence_metrics_pass1['std_ensemble_confidence']:.4f}\n")
		f.write(f"  Mean Entropy: {confidence_metrics_pass1['mean_entropy']:.4f}\n")
		f.write(f"  Mean Majority %: {confidence_metrics_pass1['mean_majority_percentage']:.1f}%\n")
		f.write(f"  Tie-resolved samples: {reasoning_pass1['tie_resolved_count']}\n")
		f.write(f"  Low-confidence samples: {reasoning_pass1['low_conf_count']}\n\n")
		
		# Confidence check results
		f.write("CONFIDENCE CHECK RESULTS\n")
		f.write("-"*80 + "\n")
		f.write(f"Confidence > 0.6? {mean_conf > 0.6} (actual: {mean_conf:.4f})\n")
		f.write(f"Entropy < 1.0? {mean_entropy < 1.0} (actual: {mean_entropy:.4f})\n")
		f.write(f"Majority > 50%? {mean_majority > 50.0} (actual: {mean_majority:.1f}%)\n")
		f.write(f"No ties? {not has_ties} (ties: {tie_count})\n")
		f.write(f"Pass 1 is CONFIDENT: {is_confident}\n")
		f.write(f"Pass 2 Executed: {pass2_executed}\n\n")
		
		# If Pass 2 was executed, show comparison
		if pass2_executed:
			f.write("PASS 2 RESULTS (Tool-Enriched Features)\n")
			f.write("-"*80 + "\n")
			f.write(f"Tool Features Added: ~50 (StandardScaler normalized)\n")
			f.write(f"Total Input Dimensions: {X_test.shape[1] + 50} (original + tools)\n\n")
			
			f.write(f"Pass 2 Confidence Metrics:\n")
			f.write(f"  Mean Confidence: {confidence_metrics_pass2['mean_ensemble_confidence']:.4f}\n")
			f.write(f"  Std Dev Confidence: {confidence_metrics_pass2['std_ensemble_confidence']:.4f}\n")
			f.write(f"  Mean Entropy: {confidence_metrics_pass2['mean_entropy']:.4f}\n")
			f.write(f"  Mean Majority %: {confidence_metrics_pass2['mean_majority_percentage']:.1f}%\n")
			f.write(f"  Tie-resolved samples: {reasoning_pass2['tie_resolved_count']}\n")
			f.write(f"  Low-confidence samples: {reasoning_pass2['low_conf_count']}\n\n")
			
			f.write("COMPARISON: PASS 1 vs PASS 2\n")
			f.write("-"*80 + "\n")
			f.write(f"Confidence Improvement: {(confidence_metrics_pass2['mean_ensemble_confidence'] - confidence_metrics_pass1['mean_ensemble_confidence']):.4f}\n")
			f.write(f"Entropy Change: {(confidence_metrics_pass2['mean_entropy'] - confidence_metrics_pass1['mean_entropy']):.4f}\n")
			f.write(f"Majority % Improvement: {(confidence_metrics_pass2['mean_majority_percentage'] - confidence_metrics_pass1['mean_majority_percentage']):.1f}%\n")
			f.write(f"Tie Resolution Improvement: {(reasoning_pass1['tie_resolved_count'] - reasoning_pass2['tie_resolved_count'])} fewer ties\n")
			f.write(f"Low-Confidence Improvement: {(reasoning_pass1['low_conf_count'] - reasoning_pass2['low_conf_count'])} fewer low-conf samples\n\n")
		
		# Overall metrics using final results
		f.write("="*80 + "\n")
		f.write("OVERALL CLASSIFICATION METRICS (Final Results)\n")
		f.write("="*80 + "\n\n")
		f.write(f"Accuracy: {acc:.4f}\n")
		f.write(f"Macro Precision: {prec_macro:.4f}\n")
		f.write(f"Macro Recall: {recall_macro:.4f}\n")
		f.write(f"Macro F1: {f1_macro:.4f}\n\n")
		f.write(f"Weighted Precision: {prec_weighted:.4f}\n")
		f.write(f"Weighted Recall: {recall_weighted:.4f}\n")
		f.write(f"Weighted F1: {f1_weighted:.4f}\n\n")
		
		# Confusion matrix
		f.write("Confusion Matrix (rows=true, cols=pred):\n")
		for row in cm:
			f.write("  " + ", ".join(str(int(x)) for x in row) + "\n")
		f.write("\nClassification Report:\n")
		f.write(report + "\n")
		
		# Prediction distribution
		f.write("\n" + "="*80 + "\n")
		f.write("PREDICTION DISTRIBUTION\n")
		f.write("="*80 + "\n\n")
		period_names = {0: '60-day', 1: '30-day', 2: '15-day', 3: '10-day'}
		pred_dist = final_confidence_metrics['prediction_distribution']
		for class_id in range(4):
			count = pred_dist[class_id]
			pct = (count / len(X_test) * 100) if len(X_test) > 0 else 0
			f.write(f"  Class {class_id} ({period_names[class_id]}): {count} samples ({pct:.1f}%)\n")
		
		# Per-sample breakdown (5 random samples)
		f.write("\n" + "="*80 + "\n")
		f.write("PER-SAMPLE DETAILED BREAKDOWN (5 random samples)\n")
		f.write("="*80 + "\n\n")
		rng = np.random.default_rng(42)
		n_show = min(5, len(X_test))
		sample_indices = rng.choice(len(X_test), size=n_show, replace=False)
		for i in sample_indices:
			f.write(f"Sample {i}:\n")
			f.write(f"  True Label: {y_test[i]} ({period_names[y_test[i]]})\n")
			f.write(f"  Ensemble Prediction: Class {final_vote_results['ensemble_predictions'][i]} ({period_names[final_vote_results['ensemble_predictions'][i]]})\n")
			f.write(f"  Confidence: {final_vote_results['ensemble_confidences'][i]:.4f}\n")
			f.write(f"  Entropy: {final_vote_results['vote_entropies'][i]:.4f}\n")
			f.write(f"  Majority %: {final_vote_results['majority_percentages'][i]:.1f}%\n")
			
			f.write(f"  Vote Counts: [")
			for c in range(4):
				f.write(f"{int(final_vote_results['vote_counts'][i, c])}")
				if c < 3:
					f.write(", ")
			f.write("]\n")
			
			f.write(f"  Aggregate Probabilities: [")
			for c in range(4):
				f.write(f"{final_vote_results['aggregate_probabilities'][i, c]:.4f}")
				if c < 3:
					f.write(", ")
			f.write("]\n")
			
			# Reasoning
			reason_row = next((r for r in final_reasoning['per_sample'] if r['index'] == i), None)
			if reason_row is not None:
				f.write(f"  Reasoning: {reason_row['method']} - {reason_row['reason_text']}\n")
			
			f.write(f"  Per-Model Predictions:\n")
			for model_name in final_all_predictions.keys():
				pred = final_all_predictions[model_name][i]
				conf = final_vote_results['per_model_confidences'][model_name][i]
				f.write(f"    {model_name}: Class {pred} (conf={conf:.4f})\n")
			f.write("\n")
		
		f.write("="*80 + "\n")
		f.write("END OF REPORT\n")
		f.write("="*80 + "\n")
	
	print(f"\nSaved voting results to {results_dir / 'ensemble_voting_results.txt'}")
	
	# Save detailed predictions to CSV
	predictions_df = pd.DataFrame({
		'sample_id': range(len(X_test)),
		'ensemble_prediction': final_vote_results['ensemble_predictions'],
		'ensemble_confidence': final_vote_results['ensemble_confidences'],
		'vote_entropy': final_vote_results['vote_entropies'],
		'majority_percentage': final_vote_results['majority_percentages'],
		'true_label': y_test,
		'p0': final_vote_results['aggregate_probabilities'][:, 0],
		'p1': final_vote_results['aggregate_probabilities'][:, 1],
		'p2': final_vote_results['aggregate_probabilities'][:, 2],
		'p3': final_vote_results['aggregate_probabilities'][:, 3],
	})
	
	predictions_df.to_csv(results_dir / 'ensemble_predictions.csv', index=False)
	print(f"Saved detailed predictions to {results_dir / 'ensemble_predictions.csv'}\n")
	
	print("="*80)
	print("EXECUTION SUMMARY")
	print("="*80)
	print(f"Pass 1 Executed: Yes")
	print(f"Pass 2 Executed: {pass2_executed}")
	print(f"Final Accuracy: {acc:.4f}")
	print(f"Final Mean Confidence: {final_confidence_metrics['mean_ensemble_confidence']:.4f}")
	print(f"Final Mean Entropy: {final_confidence_metrics['mean_entropy']:.4f}")
	print(f"Final Mean Majority %: {final_confidence_metrics['mean_majority_percentage']:.1f}%")
	print("="*80)
	
	# ========================================================================
	# SAVE ENSEMBLE PREDICTIONS TO CRITIC DATASET (ALL DATA)
	# ========================================================================
	print("\n" + "="*80)
	print("CRITIC DATASET PREPARATION")
	print("="*80)
	save_ensemble_to_critic(X, y, models)
	print("\nCritic dataset ready for model training.")


if __name__ == '__main__':
	main()

