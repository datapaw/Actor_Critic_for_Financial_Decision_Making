# Critic Agent - figures out optimal entry timing
# Takes the ensemble's prediction and decides: enter now, wait 1 day, or wait 2 days?
# Looks at:
# - ensemble confidence/agreement
# - current market conditions
# - what happens if we wait (using shifted forward returns)

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# tensorflow imports
try:
	import tensorflow as tf
	from tensorflow import keras
	from tensorflow.keras import layers
	print(f"TensorFlow {tf.__version__} loaded successfully")
except ImportError as e:
	print(f"Failed to import TensorFlow: {e}")
	tf = None
	keras = None
	layers = None
except Exception as e:
	print(f"Unexpected error importing TensorFlow: {type(e).__name__}: {e}")
	tf = None
	keras = None
	layers = None


def _require_tf():
	if tf is None:
		raise ImportError('TensorFlow is required. Install with `pip install tensorflow`.')


def load_critic_dataset(critic_csv_path=None):
	"""
	Load sp500_critic.csv which contains:
	- Original market features (Close, High, Low, Open, Volume, etc.)
	- Ensemble voting outputs (ensemble_prediction, ensemble_confidence, vote_entropy, etc.)
	- Forward return columns including _1 and _2 shifted versions
	- Strategy metrics
	
	Returns:
		X: Feature array for critic model
		y: Target labels (timing decision: 0=immediate, 1=wait 1 day, 2=wait 2 days)
		feature_cols: List of feature column names
		df: Original dataframe
	"""
	repo_root = Path(__file__).resolve().parent
	if critic_csv_path is None:
		critic_csv_path = repo_root / 'data' / 'dataset' / 'sp500_critic.csv'
	
	critic_csv_path = Path(critic_csv_path)
	if not critic_csv_path.exists():
		raise FileNotFoundError(f"Critic CSV not found at: {critic_csv_path}")
	
	df = pd.read_csv(critic_csv_path)
	
	# Parse date
	if 'Date' in df.columns:
		df['Date'] = pd.to_datetime(df['Date']).dt.date
	
	print(f"Loaded critic dataset: {len(df)} rows, {len(df.columns)} columns")
	
	return df


def compute_optimal_timing(df):
	"""
	Compute optimal timing decision based on forward returns.
	
	For each strategy class (10, 15, 30, 60 days), we have:
	- forward_return_X: return if enter immediately
	- forward_return_X_1: return if wait 1 day then hold X days
	- forward_return_X_2: return if wait 2 days then hold X days
	
	Decision logic:
	1. Compare returns for immediate vs wait 2 days
	2. Choose timing that gives highest expected return
	3. Weight by strategy confidence
	
	Returns:
		timing_labels: Array of {0, 1} for immediate/wait2days
	"""
	periods = [10, 15, 30, 60]
	n_samples = len(df)
	timing_labels = np.zeros(n_samples, dtype=int)
	
	for i in range(n_samples):
		# Get ensemble prediction (which period)
		ensemble_pred = df.loc[i, 'ensemble_prediction']
		
		# Map to period
		period_map = {0: 10, 1: 15, 2: 30, 3: 60}
		chosen_period = period_map.get(int(ensemble_pred), 30)
		
		# Get forward returns for this period
		ret_0 = df.loc[i, f'forward_return_{chosen_period}']  # immediate
		ret_2 = df.loc[i, f'forward_return_{chosen_period}_2']  # wait 2 days
		
		# Handle missing values
		if pd.isna(ret_0):
			ret_0 = 0
		if pd.isna(ret_2):
			ret_2 = 0
		
		# Choose timing: 0=immediate, 1=wait 2 days
		if ret_2 > ret_0:
			timing_labels[i] = 1  # wait 2 days
		else:
			timing_labels[i] = 0  # immediate
	
	return timing_labels


def prepare_critic_features(df):
	"""
	Prepare feature matrix for critic model.
	
	Features include ONLY information available at decision time:
	1. Ensemble voting outputs (confidence, entropy, agreements) - from current models
	2. Market conditions (price, volume) - current market state
	3. Technical indicators (RSI, MACD, Bollinger Bands, ATR, momentum, etc.)
	4. Flags and reliability metrics - decision quality indicators
	
	EXCLUDED (future information / not available at decision time):
	- forward_return_* (future returns - we don't know these yet!)
	- strategy_class_* (derived from future returns)
	- strategy_confidence_* / strategy_margin_* (redundant with ensemble metrics)
	
	Total: ~40 features (clean, non-leaking, with technical analysis)
	"""
	feature_cols = []
	
	# 1. Ensemble voting outputs (what we know NOW from the models)
	voting_cols = [
		'ensemble_prediction',
		'ensemble_confidence',
		'vote_entropy',
		'majority_percentage',
		'prediction_margin',
		'prediction_margin_normalized',
		'models_in_agreement',
		'agreement_percentage',
		'prob_class_0',
		'prob_class_1',
		'prob_class_2',
		'prob_class_3'
	]
	feature_cols.extend([c for c in voting_cols if c in df.columns])
	
	# 2. Current market conditions (current state)
	market_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
	feature_cols.extend([c for c in market_cols if c in df.columns])
	
	# 3. Technical Indicators (momentum, trend, volatility)
	technical_cols = [
		'rsi_14',                      # Overbought/oversold
		'macd_line',                   # Trend momentum
		'macd_signal',                 # Signal line
		'macd_histogram',              # MACD divergence
		'bb_upper', 'bb_lower',        # Bollinger Band levels
		'bb_width',                    # Volatility compression
		'bb_position',                 # Position within bands
		'atr_14',                      # Volatility (True Range)
		'momentum_10',                 # Rate of change
		'sma_5', 'sma_20',             # Short/medium moving averages
		'ema_12',                      # Exponential moving average
		'price_position',              # Price within 20-day range
		'volatility_20',               # Volatility measure
		'volume_sma_20',               # Average volume
		'volume_ratio'                 # Volume strength vs average
	]
	feature_cols.extend([c for c in technical_cols if c in df.columns])
	
	# 4. Decision quality flags
	flag_cols = ['had_tie', 'low_confidence_flag']
	feature_cols.extend([c for c in flag_cols if c in df.columns])
	
	# Extract features
	X_df = df[feature_cols].copy()
	
	# Convert boolean columns to int
	for col in X_df.columns:
		if X_df[col].dtype == bool:
			X_df[col] = X_df[col].astype(int)
	
	# Replace missing values
	X_df = X_df.replace(["", " ", None], np.nan)
	X_df = X_df.fillna(0)
	
	# Ensure all numeric
	for col in X_df.columns:
		X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
	
	X = X_df.values.astype(np.float32)
	
	return X, feature_cols


def create_sequences(X, y, sequence_length=10):
	"""
	Create sequences for LSTM processing.
	
	Args:
		X: Feature array (n_samples, n_features)
		y: Labels (n_samples,)
		sequence_length: Number of days to include in each sequence
	
	Returns:
		X_seq: Sequences (n_sequences, sequence_length, n_features)
		y_seq: Labels for sequences (n_sequences,) - using the last day's label
		valid_indices: Original indices corresponding to each sequence
	"""
	n_samples = len(X)
	if n_samples < sequence_length:
		raise ValueError(f"Not enough samples ({n_samples}) for sequence_length={sequence_length}")
	
	n_sequences = n_samples - sequence_length + 1
	X_seq = np.zeros((n_sequences, sequence_length, X.shape[1]), dtype=np.float32)
	y_seq = np.zeros(n_sequences, dtype=int)
	valid_indices = np.zeros(n_sequences, dtype=int)
	
	for i in range(n_sequences):
		X_seq[i] = X[i:i+sequence_length]
		y_seq[i] = y[i+sequence_length-1]  # Use last day's label
		valid_indices[i] = i + sequence_length - 1
	
	return X_seq, y_seq, valid_indices


def split_train_val_test(X, y, val_size=0.15, test_size=0.15):
	"""Split data chronologically for time series."""
	n = len(X)
	test_idx = int(n * (1 - test_size))
	val_idx = int(test_idx * (1 - val_size))
	
	X_train, y_train = X[:val_idx], y[:val_idx]
	X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
	X_test, y_test = X[test_idx:], y[test_idx:]
	
	return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_critic_model(input_shape, n_classes=2):
	"""
	Build LSTM-enhanced critic neural network for sequence processing.
	
	Architecture:
	- LSTM layers to capture temporal patterns across 10-day sequences
	- Bidirectional LSTM for forward/backward context
	- Attention mechanism to weight important timesteps
	- Dense layers for final decision making
	- Batch normalization and dropout for regularization
	
	Args:
		input_shape: Tuple (sequence_length, n_features) e.g., (10, 40)
		n_classes: Number of timing classes (default 2: immediate, wait2days)
	
	Returns:
		Compiled Keras model
	"""
	_require_tf()
	
	inp = layers.Input(shape=input_shape, name='critic_sequence_input')
	
	# =====================================================================
	# BLOCK 1: Bidirectional LSTM for temporal pattern extraction
	# =====================================================================
	x = layers.Bidirectional(
		layers.LSTM(128, return_sequences=True, name='lstm1'),
		name='bi_lstm1'
	)(inp)
	x = layers.LayerNormalization(name='ln1')(x)
	x = layers.Dropout(0.3, name='dropout1')(x)
	
	# =====================================================================
	# BLOCK 2: Second LSTM layer with residual connection
	# =====================================================================
	x = layers.Bidirectional(
		layers.LSTM(96, return_sequences=True, name='lstm2'),
		name='bi_lstm2'
	)(x)
	x = layers.LayerNormalization(name='ln2')(x)
	x = layers.Dropout(0.25, name='dropout2')(x)
	
	# =====================================================================
	# BLOCK 3: Attention mechanism to weight important timesteps
	# =====================================================================
	# Compute attention scores
	attention = layers.Dense(1, activation='tanh', name='attention_tanh')(x)
	attention = layers.Flatten(name='attention_flatten')(attention)
	attention = layers.Activation('softmax', name='attention_softmax')(attention)
	attention = layers.RepeatVector(192, name='attention_repeat')(attention)
	attention = layers.Permute([2, 1], name='attention_permute')(attention)
	
	# Apply attention weights
	x = layers.Multiply(name='attention_multiply')([x, attention])
	x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name='attention_sum')(x)
	
	# =====================================================================
	# BLOCK 4: Dense processing of attended features
	# =====================================================================
	x = layers.Dense(128, activation='relu', name='dense1')(x)
	x = layers.BatchNormalization(name='bn1')(x)
	x = layers.Dropout(0.3, name='dropout3')(x)
	
	# =====================================================================
	# BLOCK 5: Deeper feature extraction with residual
	# =====================================================================
	residual = layers.Dense(96, activation='relu', name='residual_1')(x)
	x = layers.Dense(96, activation='relu', name='dense2')(x)
	x = layers.BatchNormalization(name='bn2')(x)
	x = layers.Add(name='add1')([x, residual])
	x = layers.Activation('relu', name='relu_after_add1')(x)
	x = layers.Dropout(0.2, name='dropout4')(x)
	
	# =====================================================================
	# BLOCK 6: Final decision layers
	# =====================================================================
	x = layers.Dense(64, activation='relu', name='dense3')(x)
	x = layers.BatchNormalization(name='bn3')(x)
	x = layers.Dropout(0.15, name='dropout5')(x)
	
	x = layers.Dense(32, activation='relu', name='dense4')(x)
	x = layers.BatchNormalization(name='bn4')(x)
	
	# =====================================================================
	# OUTPUT: Timing decision (2 classes: immediate or wait 2 days)
	# =====================================================================
	out = layers.Dense(n_classes, activation='softmax', name='timing_output')(x)
	
	model = keras.Model(inputs=inp, outputs=out, name='critic_model_lstm')
	model.compile(
		optimizer=keras.optimizers.Adam(learning_rate=0.0003),
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)
	
	return model


def train_critic(model, X_train, y_train, X_val, y_val, epochs=150, batch_size=16, save_path='models/critic_best_now_or_2.keras'):
	"""
	Train critic model with enhanced callbacks and regularization.
	"""
	_require_tf()
	
	save_path = Path(save_path)
	save_path.parent.mkdir(parents=True, exist_ok=True)
	
	callbacks = [
		keras.callbacks.EarlyStopping(
			monitor='val_loss',
			patience=20,
			restore_best_weights=True,
			verbose=1
		),
		keras.callbacks.ModelCheckpoint(
			filepath=str(save_path),
			monitor='val_accuracy',
			save_best_only=True,
			verbose=1
		),
		keras.callbacks.ReduceLROnPlateau(
			monitor='val_loss',
			factor=0.5,
			patience=10,
			min_lr=1e-7,
			verbose=1
		),
		keras.callbacks.EarlyStopping(
			monitor='val_accuracy',
			patience=20,
			restore_best_weights=True,
			verbose=1,
			mode='max'
		)
	]
	
	print("\n" + "="*80)
	print("Training Enhanced Critic Model")
	print("="*80)
	
	history = model.fit(
		X_train, y_train,
		validation_data=(X_val, y_val),
		epochs=epochs,
		batch_size=batch_size,
		callbacks=callbacks,
		verbose=1
	)
	
	print(f"\nBest validation accuracy: {max(history.history['val_accuracy']):.4f}")
	print(f"Model saved to: {save_path}")
	
	return history


def evaluate_critic(model, X_test, y_test):
	"""
	Evaluate critic model performance.
	"""
	_require_tf()
	
	print("\n" + "="*80)
	print("Evaluating Critic Model")
	print("="*80)
	
	# Predictions
	probs = model.predict(X_test, verbose=0)
	y_pred = np.argmax(probs, axis=1)
	
	# Metrics
	accuracy = accuracy_score(y_test, y_pred)
	print(f"\nTest Accuracy: {accuracy:.4f}")
	
	print("\nClassification Report:")
	print(classification_report(y_test, y_pred, target_names=['Immediate', 'Wait 2 Days']))
	
	print("\nConfusion Matrix:")
	cm = confusion_matrix(y_test, y_pred)
	print(cm)
	
	# Timing distribution
	print("\nPrediction Distribution:")
	unique, counts = np.unique(y_pred, return_counts=True)
	for timing, count in zip(unique, counts):
		timing_names = {0: 'Immediate', 1: 'Wait 2 Days'}
		print(f"  {timing_names.get(timing, f'Class {timing}')}: {count} ({count/len(y_pred)*100:.1f}%)")
	
	print("\nActual Distribution:")
	unique, counts = np.unique(y_test, return_counts=True)
	for timing, count in zip(unique, counts):
		timing_names = {0: 'Immediate', 1: 'Wait 2 Days'}
		print(f"  {timing_names.get(timing, f'Class {timing}')}: {count} ({count/len(y_test)*100:.1f}%)")
	
	# Confidence analysis
	print("\nConfidence Analysis:")
	mean_conf = np.max(probs, axis=1).mean()
	print(f"  Mean prediction confidence: {mean_conf:.4f}")
	
	# Per-class confidence
	for cls in range(2):
		cls_mask = (y_pred == cls)
		if cls_mask.sum() > 0:
			cls_conf = np.max(probs[cls_mask], axis=1).mean()
			timing_names = {0: 'Immediate', 1: 'Wait 2 Days'}
			print(f"  {timing_names.get(cls, f'Class {cls}')} confidence: {cls_conf:.4f}")
	
	return {
		'accuracy': accuracy,
		'predictions': y_pred,
		'probabilities': probs,
		'confusion_matrix': cm
	}


def save_critic_results(df, y_pred, probs, output_path='results/critic_timing_results_v2.csv'):
	"""
	Save critic predictions to CSV.
	"""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	
	# Create results dataframe
	results_df = df.copy()
	results_df['critic_timing_decision'] = y_pred
	results_df['critic_timing_prob_0'] = probs[:, 0]
	results_df['critic_timing_prob_1'] = probs[:, 1]
	results_df['critic_confidence'] = np.max(probs, axis=1)
	
	# Add timing labels
	timing_names = {0: 'immediate', 1: 'wait_2_days'}
	results_df['critic_timing_label'] = results_df['critic_timing_decision'].map(timing_names)
	
	results_df.to_csv(output_path, index=False)
	print(f"\nCritic results saved to: {output_path}")
	
	return results_df


def save_evaluation_report(results, y_test, output_path='results/critic_evaluation_v2.txt'):
	"""
	Save detailed evaluation metrics and confusion matrix to text file.
	
	Args:
		results: Dict from evaluate_critic() containing accuracy, predictions, probabilities, confusion_matrix
		y_test: True labels for test set
		output_path: Path to save report
	"""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	
	accuracy = results['accuracy']
	y_pred = results['predictions']
	probs = results['probabilities']
	cm = results['confusion_matrix']
	
	with open(output_path, 'w') as f:
		f.write("="*80 + "\n")
		f.write("CRITIC MODEL EVALUATION REPORT\n")
		f.write("="*80 + "\n\n")
		
		# Overall metrics
		f.write("OVERALL METRICS\n")
		f.write("-" * 80 + "\n")
		f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
		
		# Classification report
		f.write("CLASSIFICATION REPORT\n")
		f.write("-" * 80 + "\n")
		f.write(classification_report(y_test, y_pred, target_names=['Immediate', 'Wait 2 Days']))
		f.write("\n")
		
		# Confusion matrix
		f.write("CONFUSION MATRIX\n")
		f.write("-" * 80 + "\n")
		f.write("Columns: Predicted | Rows: Actual\n")
		f.write("Classes: [Immediate, Wait 2 Days]\n\n")
		f.write(str(cm) + "\n\n")
		
		# Formatted confusion matrix
		f.write("Confusion Matrix (Formatted):\n")
		f.write("                  Pred:Immed  Pred:Wait2\n")
		class_names = ['Actual:Immed', 'Actual:Wait2']
		for i, class_name in enumerate(class_names):
			f.write(f"{class_name:20s} {cm[i, 0]:10d}  {cm[i, 1]:10d}\n")
		f.write("\n")
		
		# Prediction distribution
		f.write("PREDICTION DISTRIBUTION\n")
		f.write("-" * 80 + "\n")
		unique, counts = np.unique(y_pred, return_counts=True)
		for timing, count in zip(unique, counts):
			timing_names = {0: 'Immediate', 1: 'Wait 2 Days'}
			percentage = count / len(y_pred) * 100
			f.write(f"{timing_names.get(timing, f'Class {timing}'):20s}: {count:6d} ({percentage:5.1f}%)\n")
		f.write("\n")
		
		# Actual distribution
		f.write("ACTUAL DISTRIBUTION (Test Set)\n")
		f.write("-" * 80 + "\n")
		unique, counts = np.unique(y_test, return_counts=True)
		for timing, count in zip(unique, counts):
			timing_names = {0: 'Immediate', 1: 'Wait 2 Days'}
			percentage = count / len(y_test) * 100
			f.write(f"{timing_names.get(timing, f'Class {timing}'):20s}: {count:6d} ({percentage:5.1f}%)\n")
		f.write("\n")
		
		# Confidence analysis
		f.write("CONFIDENCE ANALYSIS\n")
		f.write("-" * 80 + "\n")
		mean_conf = np.max(probs, axis=1).mean()
		f.write(f"Mean prediction confidence: {mean_conf:.4f}\n\n")
		
		f.write("Per-Class Confidence:\n")
		for cls in range(2):
			cls_mask = (y_pred == cls)
			if cls_mask.sum() > 0:
				cls_conf = np.max(probs[cls_mask], axis=1).mean()
				timing_names = {0: 'Immediate', 1: 'Wait 2 Days'}
				f.write(f"  {timing_names.get(cls, f'Class {cls}')}: {cls_conf:.4f}\n")
		f.write("\n")
		
		# Per-class metrics from confusion matrix
		f.write("PER-CLASS METRICS (from Confusion Matrix)\n")
		f.write("-" * 80 + "\n")
		timing_names = {0: 'Immediate', 1: 'Wait 2 Days'}
		
		for cls in range(2):
			tp = cm[cls, cls]
			fp = cm[:, cls].sum() - tp
			fn = cm[cls, :].sum() - tp
			tn = cm.sum() - tp - fp - fn
			
			precision = tp / (tp + fp) if (tp + fp) > 0 else 0
			recall = tp / (tp + fn) if (tp + fn) > 0 else 0
			f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
			
			f.write(f"\n{timing_names.get(cls, f'Class {cls}')}:\n")
			f.write(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n")
			f.write(f"  Precision: {precision:.4f}\n")
			f.write(f"  Recall: {recall:.4f}\n")
			f.write(f"  F1-Score: {f1:.4f}\n")
		
		f.write("\n" + "="*80 + "\n")
	
	print(f"\nEvaluation report saved to: {output_path}")


def analyze_critic_recommendations(y_pred, y_test, probs, output_path='results/critic_recommendation_summary.csv'):
	"""
	Analyze and save critic recommendation summary to CSV.
	
	Determines whether critic recommends immediate entry or waiting,
	and compares with optimal timing based on actual returns.
	"""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	
	# Count predictions
	immediate_count = int(np.sum(y_pred == 0))
	wait2_count = int(np.sum(y_pred == 1))
	total = len(y_pred)
	
	# Count optimal (actual)
	immediate_optimal = int(np.sum(y_test == 0))
	wait2_optimal = int(np.sum(y_test == 1))
	
	# Determine recommendation
	if immediate_count > wait2_count:
		recommendation = "Enter IMMEDIATELY"
		recommendation_pct = immediate_count / total * 100
	elif wait2_count > immediate_count:
		recommendation = "WAIT 2 DAYS"
		recommendation_pct = wait2_count / total * 100
	else:
		recommendation = "BALANCED (50-50)"
		recommendation_pct = 50.0
	
	# Determine optimal strategy
	if immediate_optimal > wait2_optimal:
		optimal_strategy = "Immediate entry"
		optimal_pct = immediate_optimal / len(y_test) * 100
	elif wait2_optimal > immediate_optimal:
		optimal_strategy = "Wait 2 days"
		optimal_pct = wait2_optimal / len(y_test) * 100
	else:
		optimal_strategy = "Both equal"
		optimal_pct = 50.0
	
	# Confidence stats
	mean_conf = float(np.max(probs, axis=1).mean())
	immediate_conf = float(np.max(probs[y_pred == 0], axis=1).mean()) if immediate_count > 0 else 0.0
	wait2_conf = float(np.max(probs[y_pred == 1], axis=1).mean()) if wait2_count > 0 else 0.0
	
	# Create summary dataframe
	summary_data = [
		{"Metric": "Critic Recommendation", "Value": recommendation, "Percentage": f"{recommendation_pct:.1f}%"},
		{"Metric": "Immediate Predictions", "Value": immediate_count, "Percentage": f"{immediate_count/total*100:.1f}%"},
		{"Metric": "Wait 2 Days Predictions", "Value": wait2_count, "Percentage": f"{wait2_count/total*100:.1f}%"},
		{"Metric": "Total Test Samples", "Value": total, "Percentage": "100.0%"},
		{"Metric": "", "Value": "", "Percentage": ""},
		{"Metric": "Optimal Strategy (Actual)", "Value": optimal_strategy, "Percentage": f"{optimal_pct:.1f}%"},
		{"Metric": "Optimal: Immediate", "Value": immediate_optimal, "Percentage": f"{immediate_optimal/len(y_test)*100:.1f}%"},
		{"Metric": "Optimal: Wait 2 Days", "Value": wait2_optimal, "Percentage": f"{wait2_optimal/len(y_test)*100:.1f}%"},
		{"Metric": "", "Value": "", "Percentage": ""},
		{"Metric": "Mean Confidence", "Value": f"{mean_conf:.4f}", "Percentage": ""},
		{"Metric": "Immediate Confidence", "Value": f"{immediate_conf:.4f}", "Percentage": ""},
		{"Metric": "Wait 2 Days Confidence", "Value": f"{wait2_conf:.4f}", "Percentage": ""},
	]
	
	summary_df = pd.DataFrame(summary_data)
	summary_df.to_csv(output_path, index=False)
	
	# Print summary
	print("\n" + "="*80)
	print("CRITIC RECOMMENDATION SUMMARY")
	print("="*80)
	print(f"\nRECOMMENDATION: {recommendation}")
	print(f"  Confidence in recommendation: {recommendation_pct:.1f}%")
	print(f"\nPrediction Breakdown:")
	print(f"  Immediate: {immediate_count} samples ({immediate_count/total*100:.1f}%)")
	print(f"  Wait 2 days: {wait2_count} samples ({wait2_count/total*100:.1f}%)")
	print(f"\nOptimal Strategy (based on actual returns):")
	print(f"  {optimal_strategy} ({optimal_pct:.1f}% of cases)")
	print(f"\nMean prediction confidence: {mean_conf:.4f}")
	print("="*80)
	print(f"\nRecommendation summary saved to: {output_path}")
	
	return summary_df


def main():
	"""
	Main execution:
	1. Load critic dataset (sp500_critic.csv with ensemble outputs)
	2. Compute optimal timing labels
	3. Prepare features
	4. Train critic model
	5. Evaluate and save results
	"""
	print("="*80)
	print("CRITIC AGENT: Trading Timing Decision")
	print("="*80)
	
	# Load data
	df = load_critic_dataset()
	
	# Compute optimal timing labels
	print("\nComputing optimal timing labels...")
	y = compute_optimal_timing(df)
	print(f"Timing distribution:")
	unique, counts = np.unique(y, return_counts=True)
	for timing, count in zip(unique, counts):
		timing_names = {0: 'Immediate', 1: 'Wait 2 Days'}
		print(f"  {timing_names.get(timing, f'Class {timing}')}: {count} ({count/len(y)*100:.1f}%)")
	
	# Prepare features
	print("\nPreparing critic features...")
	X, feature_cols = prepare_critic_features(df)
	print(f"Feature matrix shape: {X.shape}")
	print(f"Number of features: {len(feature_cols)}")
	print(f"Sample features: {feature_cols[:10]}")
	
	# Normalize features BEFORE creating sequences
	print("\nNormalizing features...")
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	
	# Create sequences for LSTM (10-day windows)
	print("\nCreating sequences (10-day windows)...")
	sequence_length = 10
	X_seq, y_seq, valid_indices = create_sequences(X_scaled, y, sequence_length=sequence_length)
	print(f"Sequences created: {len(X_seq)} sequences of shape {X_seq.shape}")
	print(f"Each sequence contains {sequence_length} days of {X_seq.shape[2]} features")
	
	# Split data
	print("\nSplitting data...")
	(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_train_val_test(X_seq, y_seq)
	print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
	
	# Build model with sequence input shape
	print("\nBuilding LSTM critic model...")
	input_shape = (sequence_length, X.shape[1])  # (10, n_features)
	model = build_critic_model(input_shape=input_shape, n_classes=2)
	print(model.summary())
	
	# Train model
	history = train_critic(
		model, X_train, y_train, X_val, y_val,
		epochs=150,
		batch_size=16,
		save_path='models/critic_lstm_now_or_2.keras'
	)
	
	# Evaluate
	results = evaluate_critic(model, X_test, y_test)
	
	# Save evaluation report
	save_evaluation_report(results, y_test, output_path='results/critic_evaluation_v2.txt')
	
	# Analyze and save recommendations
	analyze_critic_recommendations(results['predictions'], y_test, results['probabilities'],
	                                output_path='results/critic_recommendation_summary.csv')
	
	# Save predictions on full dataset
	print("\nGenerating predictions on full dataset...")
	probs_seq = model.predict(X_seq, verbose=0)
	y_pred_seq = np.argmax(probs_seq, axis=1)
	
	# Map predictions back to original indices
	# For samples not in sequences, use neutral prediction or copy from nearest
	probs_full = np.zeros((len(df), 2), dtype=np.float32)
	y_pred_full = np.zeros(len(df), dtype=int)
	
	# Fill in predictions for samples that have sequences
	for i, orig_idx in enumerate(valid_indices):
		probs_full[orig_idx] = probs_seq[i]
		y_pred_full[orig_idx] = y_pred_seq[i]
	
	# For first sequence_length-1 samples, copy from first available prediction
	for i in range(sequence_length - 1):
		probs_full[i] = probs_seq[0]
		y_pred_full[i] = y_pred_seq[0]
	
	print(f"Predictions generated for {len(df)} samples (using 10-day sequence averaging)")
	
	results_df = save_critic_results(
		df, y_pred_full, probs_full,
		output_path='results/critic_timing_results_lstm_v2.csv'
	)
	
	# Overall recommendation for full dataset
	immediate_total = int(np.sum(y_pred_full == 0))
	wait2_total = int(np.sum(y_pred_full == 1))
	full_summary_data = [
		{"Metric": "Overall Recommendation", "Value": "Enter IMMEDIATELY" if immediate_total > wait2_total else ("WAIT 2 DAYS" if wait2_total > immediate_total else "BALANCED"), "Count": ""},
		{"Metric": "Immediate Predictions", "Value": immediate_total, "Count": f"{immediate_total/len(y_pred_full)*100:.1f}%"},
		{"Metric": "Wait 2 Days Predictions", "Value": wait2_total, "Count": f"{wait2_total/len(y_pred_full)*100:.1f}%"},
		{"Metric": "Total Samples", "Value": len(y_pred_full), "Count": "100.0%"},
	]
	full_summary_df = pd.DataFrame(full_summary_data)
	full_summary_path = Path(__file__).resolve().parent / 'results' / 'critic_overall_recommendation.csv'
	full_summary_df.to_csv(full_summary_path, index=False)
	print(f"\nOverall recommendation saved to: {full_summary_path}")
	
	print("\n" + "="*80)
	print("CRITIC TRAINING COMPLETE (LSTM with 10-day sequences)")
	print("="*80)
	print(f"Test Accuracy: {results['accuracy']:.4f}")
	print(f"Model saved: models/critic_lstm_v2.keras")
	print(f"Results saved: results/critic_timing_results_lstm_v2.csv")
	print(f"Report saved: results/critic_evaluation_report_lstm_v2.txt")
	print(f"Sequence length: {sequence_length} days")
	print(f"Total sequences processed: {len(X_seq)}")


if __name__ == '__main__':
	main()
