"""
Critic Agent for Trading Decision Timing

This critic takes ensemble voting outputs from agent_main_loop.py and original
market features to make a refined decision: should we enter the position
immediately (day 0), wait 1 day, or wait 2 days?

The critic analyzes:
1. Ensemble voting results (predictions, confidence, entropy, agreements)
2. Market conditions (price, volume, volatility features)
3. Forward return patterns (_1 and _2 shifted columns)
4. Strategy strength metrics (confidence, margins)

Output: timing_decision in {0, 1} representing immediate or wait 2 days before entering
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# TensorFlow / Keras
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


def split_train_val_test(X, y, val_size=0.15, test_size=0.15):
	"""Split data chronologically for time series."""
	n = len(X)
	test_idx = int(n * (1 - test_size))
	val_idx = int(test_idx * (1 - val_size))
	
	X_train, y_train = X[:val_idx], y[:val_idx]
	X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
	X_test, y_test = X[test_idx:], y[test_idx:]
	
	return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_critic_model(input_dim, n_classes=2):
	"""
	Build enhanced critic neural network with better architecture.
	
	Architecture:
	- Feature processing blocks with residual connections
	- Batch normalization throughout
	- Strategic dropout for regularization
	- Attention-like feature weighting
	- Deep network to capture complex patterns
	
	Args:
		input_dim: Number of input features (~40)
		n_classes: Number of timing classes (default 2: immediate, wait2days)
	
	Returns:
		Compiled Keras model
	"""
	_require_tf()
	
	inp = layers.Input(shape=(input_dim,), name='critic_input')
	
	# =====================================================================
	# BLOCK 1: Initial feature processing
	# =====================================================================
	x = layers.Dense(256, activation='relu', name='dense1')(inp)
	x = layers.BatchNormalization(name='bn1')(x)
	x = layers.Dropout(0.3, name='dropout1')(x)
	
	# =====================================================================
	# BLOCK 2: Feature combination with residual path
	# =====================================================================
	residual = layers.Dense(192, activation='relu', name='residual_1')(inp)
	x = layers.Dense(192, activation='relu', name='dense2')(x)
	x = layers.BatchNormalization(name='bn2')(x)
	x = layers.Add(name='add1')([x, residual])
	x = layers.Activation('relu', name='relu_after_add1')(x)
	x = layers.Dropout(0.25, name='dropout2')(x)
	
	# =====================================================================
	# BLOCK 3: Technical indicator processing
	# =====================================================================
	x = layers.Dense(128, activation='relu', name='dense3')(x)
	x = layers.BatchNormalization(name='bn3')(x)
	x = layers.Dropout(0.2, name='dropout3')(x)
	
	# =====================================================================
	# BLOCK 4: Deeper feature extraction with residual
	# =====================================================================
	residual2 = layers.Dense(96, activation='relu', name='residual_2')(x)
	x = layers.Dense(96, activation='relu', name='dense4')(x)
	x = layers.BatchNormalization(name='bn4')(x)
	x = layers.Add(name='add2')([x, residual2])
	x = layers.Activation('relu', name='relu_after_add2')(x)
	x = layers.Dropout(0.15, name='dropout4')(x)
	
	# =====================================================================
	# BLOCK 5: Pattern recognition
	# =====================================================================
	x = layers.Dense(64, activation='relu', name='dense5')(x)
	x = layers.BatchNormalization(name='bn5')(x)
	x = layers.Dropout(0.1, name='dropout5')(x)
	
	# =====================================================================
	# BLOCK 6: Decision bottleneck
	# =====================================================================
	x = layers.Dense(32, activation='relu', name='dense6')(x)
	x = layers.BatchNormalization(name='bn6')(x)
	
	# =====================================================================
	# OUTPUT: Timing decision (2 classes: immediate or wait 2 days)
	# =====================================================================
	out = layers.Dense(n_classes, activation='softmax', name='timing_output')(x)
	
	model = keras.Model(inputs=inp, outputs=out, name='critic_model_enhanced')
	model.compile(
		optimizer=keras.optimizers.Adam(learning_rate=0.0005),
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


def save_critic_results(df, y_pred, probs, output_path='results/critic_timing_results_now_or_2.csv'):
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


def save_evaluation_report(results, y_test, output_path='results/critic_evaluation_report_now_or_2.txt'):
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
	
	# Split data
	print("\nSplitting data...")
	(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_train_val_test(X, y)
	print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
	
	# Normalize features
	print("\nNormalizing features...")
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_val = scaler.transform(X_val)
	X_test = scaler.transform(X_test)
	
	# Build model
	print("\nBuilding critic model...")
	model = build_critic_model(input_dim=X_train.shape[1], n_classes=2)
	print(model.summary())
	
	# Train model
	history = train_critic(
		model, X_train, y_train, X_val, y_val,
		epochs=150,
		batch_size=16,
		save_path='models/critic__now_or_2.keras'
	)
	
	# Evaluate
	results = evaluate_critic(model, X_test, y_test)
	
	# Save evaluation report
	save_evaluation_report(results, y_test, output_path='results/critic_evaluation_report_now_or_2.txt')
	
	# Save predictions on full dataset
	print("\nGenerating predictions on full dataset...")
	X_full = scaler.transform(X)
	probs_full = model.predict(X_full, verbose=0)
	y_pred_full = np.argmax(probs_full, axis=1)
	
	results_df = save_critic_results(
		df, y_pred_full, probs_full,
		output_path='results/critic_timing_results_now_or_2.csv'
	)
	
	print("\n" + "="*80)
	print("CRITIC TRAINING COMPLETE")
	print("="*80)
	print(f"Test Accuracy: {results['accuracy']:.4f}")
	print(f"Model saved: models/critic_best_now_or_2.keras")
	print(f"Results saved: results/critic_timing_results_now_or_2.csv")
	print(f"Report saved: results/critic_evaluation_report_now_or_2.txt")


if __name__ == '__main__':
	main()
