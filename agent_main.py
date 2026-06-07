# Main data loader for the ensemble models
# Reads the merged dataset and prepares everything for training

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


# import tensorflow stuff - might fail if not installed
try:
	import tensorflow as tf
	from tensorflow import keras
	from tensorflow.keras import layers
except Exception:
	tf = None
	keras = None
	layers = None
	# note: will throw error later if someone tries to use models


def _require_tf():
	if tf is None:
		raise ImportError('TensorFlow not found - install with: pip install tensorflow')


def split_train_val_test(X, y, val_size=0.15, test_size=0.15, random_seed=42):
	# split chronologically - can't shuffle time series data!
	n = len(X)
	test_idx = int(n * (1 - test_size))
	val_idx = int(test_idx * (1 - val_size))
	
	X_train, y_train = X[:val_idx], y[:val_idx]
	X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
	X_test, y_test = X[test_idx:], y[test_idx:]
	
	return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_merged_dataset(path: str | Path = None):
	repo_root = Path(__file__).resolve().parent
	if path is None:
		path = repo_root / 'data' / 'dataset' / 'sp500_features_prices_merged.csv'
	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(f"Merged CSV not found at: {path}")


	df = pd.read_csv(path)

	# detect Date column (keep it if present)
	date_col = None
	for cand in ('Date', 'date', 'date_only'):
		if cand in df.columns:
			date_col = cand
			break

	if date_col is not None:
		df[date_col] = pd.to_datetime(df[date_col])
		# normalize
		df['Date'] = df[date_col].dt.date
	elif 'Date' not in df.columns:
		# do nothing if no date
		pass

	if 'target_period' not in df.columns:
		raise ValueError('`target_period` column not found in merged CSV')

	# All columns except Date and target_period are inputs
	input_cols = [c for c in df.columns if c not in ('Date', 'target_period')]

	X_df = df[input_cols].copy()
	# Replace any remaining missing-like values and coerce to numeric
	X_df = X_df.replace(["", " ", None], np.nan)
	X_df = X_df.fillna(0)
	for col in X_df.columns:
		X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)

	y = pd.to_numeric(df['target_period'], errors='coerce')
	if y.isna().any():
		raise ValueError('Some target_period values could not be parsed as integers')

	X = X_df.values.astype(np.float32)
	y = y.astype(int).values

	dates = df['Date'] if 'Date' in df.columns else None

	return X, y, input_cols, dates


def train_all_models(models, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, save_dir='models'):
	"""Train all models and save them."""
	_require_tf()
	save_path = Path(save_dir)
	save_path.mkdir(parents=True, exist_ok=True)
	
	histories = {}
	for name, model in models.items():
		print(f"\nTraining {name}...")
		
		# Callbacks: early stopping and model checkpoint
		callbacks = [
			keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
			keras.callbacks.ModelCheckpoint(
				filepath=str(save_path / f"{name}_best.keras"),
				monitor='val_accuracy',
				save_best_only=True,
				verbose=0
			),
			keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
		]
		
		hist = model.fit(
			X_train, y_train,
			validation_data=(X_val, y_val),
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks,
			verbose=1
		)
		
		histories[name] = hist.history
		print(f"  Best val_accuracy: {max(hist.history['val_accuracy']):.4f}")
		
		# Save final model
		model.save(save_path / f"{name}_final.keras")
	
	return histories


if __name__ == '__main__':
	X, y, cols, dates = load_merged_dataset()
	print('Loaded merged dataset:')
	print('  rows=', X.shape[0], 'cols=', X.shape[1])
	print('  sample input cols=', cols[:8])
	if dates is not None:
		print('  first date=', dates.iloc[0])

	# --- Model builders ---
	# Why: strong non-linear baseline; stable and fast to train
	def build_mlp(input_dim, n_classes=4):
		_require_tf()
		inp = layers.Input(shape=(input_dim,))
		x = layers.Dense(256, activation='relu')(inp)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(0.3)(x)
		x = layers.Dense(128, activation='relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(0.2)(x)
		x = layers.Dense(64, activation='relu')(x)
		out = layers.Dense(n_classes, activation='softmax')(x)
		m = keras.Model(inputs=inp, outputs=out, name='mlp')
		m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		return m

	# Why: combines a shallow wide path (memorization) with deep path (interaction learning)
	def build_widedeep(input_dim, n_classes=4):
		_require_tf()
		inp = layers.Input(shape=(input_dim,))
		wide = layers.Dense(64, activation='relu')(inp)
		deep = layers.Dense(256, activation='relu')(inp)
		deep = layers.Dense(128, activation='relu')(deep)
		x = layers.Concatenate()([wide, deep])
		x = layers.Dense(64, activation='relu')(x)
		out = layers.Dense(n_classes, activation='softmax')(x)
		m = keras.Model(inputs=inp, outputs=out, name='wide_deep')
		m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		return m

	# Why: residual connections help deeper MLP avoid vanishing gradients
	def build_resnet_mlp(input_dim, n_classes=4):
		_require_tf()
		inp = layers.Input(shape=(input_dim,))
		x = layers.Dense(128, activation='relu')(inp)
		for i in range(2):
			y = layers.Dense(128, activation='relu')(x)
			y = layers.Dense(128, activation=None)(y)
			x = layers.Add()([x, y])
			x = layers.Activation('relu')(x)
		x = layers.Dense(64, activation='relu')(x)
		out = layers.Dense(n_classes, activation='softmax')(x)
		m = keras.Model(inputs=inp, outputs=out, name='resnet_mlp')
		m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		return m

	# Why: 1D conv captures local patterns/correlations across adjacent engineered features
	def build_conv1d_model(input_dim, n_classes=4):
		_require_tf()
		inp = layers.Input(shape=(input_dim,))
		x = layers.Reshape((input_dim, 1))(inp)
		x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
		x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
		x = layers.GlobalMaxPool1D()(x)
		x = layers.Dense(64, activation='relu')(x)
		out = layers.Dense(n_classes, activation='softmax')(x)
		m = keras.Model(inputs=inp, outputs=out, name='conv1d')
		m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		return m

	# Why: squeeze-excitation reweights features per-sample to focus on informative signals
	def build_se_mlp(input_dim, n_classes=4, se_ratio=8):
		_require_tf()
		inp = layers.Input(shape=(input_dim,))
		x = layers.Dense(256, activation='relu')(inp)
		x = layers.Dense(128, activation='relu')(x)
		# Squeeze-and-excitation style gating: produce weights matching the dense block size
		se_units = 128
		se = layers.Dense(max(1, input_dim // se_ratio), activation='relu')(inp)
		se = layers.Dense(se_units, activation='sigmoid')(se)
		x = layers.Multiply()([x, se])
		x = layers.Dense(64, activation='relu')(x)
		out = layers.Dense(n_classes, activation='softmax')(x)
		m = keras.Model(inputs=inp, outputs=out, name='se_mlp')
		m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		return m

	# Why: BiLSTM + attention can model sequential structure over ordered features or lags
	def build_bilstm_att(input_dim, n_classes=4):
		_require_tf()
		inp = layers.Input(shape=(input_dim,))
		x = layers.Reshape((input_dim, 1))(inp)
		x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
		x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
		# Simple additive attention: project to scalar weights
		att_weights = layers.Dense(1, activation='tanh')(x)
		att_weights = layers.Softmax(axis=1)(att_weights)
		x = layers.Multiply()([x, att_weights])
		x = layers.GlobalAveragePooling1D()(x)
		x = layers.Dense(64, activation='relu')(x)
		x = layers.Dropout(0.2)(x)
		out = layers.Dense(n_classes, activation='softmax')(x)
		m = keras.Model(inputs=inp, outputs=out, name='bilstm_att')
		m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		return m

	# Why: stacked GRU with LayerNorm stabilizes training on longer feature sequences
	def build_stacked_gru_ln(input_dim, n_classes=4):
		_require_tf()
		inp = layers.Input(shape=(input_dim,))
		x = layers.Reshape((input_dim, 1))(inp)
		x = layers.GRU(96, return_sequences=True)(x)
		x = layers.LayerNormalization()(x)
		x = layers.GRU(64, return_sequences=False)(x)
		x = layers.Dense(64, activation='relu')(x)
		x = layers.Dropout(0.3)(x)
		out = layers.Dense(n_classes, activation='softmax')(x)
		m = keras.Model(inputs=inp, outputs=out, name='stacked_gru_ln')
		m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		return m

	# Why: transformer encoder captures pairwise feature relations via self-attention
	def build_transformer_enc(input_dim, n_classes=4, num_heads=4, d_model=64, ff_dim=128):
		_require_tf()
		inp = layers.Input(shape=(input_dim,))
		x = layers.Reshape((input_dim, 1))(inp)
		x = layers.Dense(d_model)(x)
		attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
		x = layers.Add()([x, attn_out])
		x = layers.LayerNormalization()(x)
		ff = layers.Dense(ff_dim, activation='relu')(x)
		ff = layers.Dense(d_model)(ff)
		x = layers.Add()([x, ff])
		x = layers.LayerNormalization()(x)
		x = layers.GlobalAveragePooling1D()(x)
		x = layers.Dense(64, activation='relu')(x)
		x = layers.Dropout(0.2)(x)
		out = layers.Dense(n_classes, activation='softmax')(x)
		m = keras.Model(inputs=inp, outputs=out, name='transformer_enc')
		m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		return m
	# Why: smaller MLP for regularization and speed on limited data
	def build_mlp_small(input_dim, n_classes=4):
		_require_tf()
		inp = layers.Input(shape=(input_dim,))
		x = layers.Dense(128, activation='relu')(inp)
		x = layers.Dropout(0.25)(x)
		x = layers.Dense(64, activation='relu')(x)
		x = layers.Dropout(0.25)(x)
		out = layers.Dense(n_classes, activation='softmax')(x)
		m = keras.Model(inputs=inp, outputs=out, name='mlp_small')
		m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		return m

	# Why: wider MLP to model richer interactions when data supports larger capacity
	def build_mlp_wide(input_dim, n_classes=4):
		_require_tf()
		inp = layers.Input(shape=(input_dim,))
		x = layers.Dense(512, activation='relu')(inp)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(0.3)(x)
		x = layers.Dense(256, activation='relu')(x)
		x = layers.BatchNormalization()(x)
		x = layers.Dropout(0.3)(x)
		x = layers.Dense(128, activation='relu')(x)
		x = layers.Dropout(0.2)(x)
		out = layers.Dense(n_classes, activation='softmax')(x)
		m = keras.Model(inputs=inp, outputs=out, name='mlp_wide')
		m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		return m

	# Why: residual Conv1D stack with squeeze-excite for stronger local pattern modeling
	def build_resnet_conv1d(input_dim, n_classes=4):
		_require_tf()
		inp = layers.Input(shape=(input_dim,))
		x = layers.Reshape((input_dim, 1))(inp)
		def res_block(t, filters):
			shortcut = t
			y = layers.Conv1D(filters, 3, padding='same', activation='relu')(t)
			y = layers.Conv1D(filters, 3, padding='same', activation=None)(y)
			# squeeze-excite
			se = layers.GlobalAveragePooling1D()(y)
			se = layers.Dense(filters // 4, activation='relu')(se)
			se = layers.Dense(filters, activation='sigmoid')(se)
			y = layers.Multiply()([y, se])
			y = layers.Add()([shortcut, y])
			y = layers.Activation('relu')(y)
			return y
		x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
		x = res_block(x, 64)
		x = res_block(x, 64)
		x = layers.GlobalAveragePooling1D()(x)
		x = layers.Dense(64, activation='relu')(x)
		x = layers.Dropout(0.2)(x)
		out = layers.Dense(n_classes, activation='softmax')(x)
		m = keras.Model(inputs=inp, outputs=out, name='resnet_conv1d')
		m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		return m

	def build_all_models(input_dim, n_classes=4):
		models = {}
		models['mlp'] = build_mlp(input_dim, n_classes)
		models['wide_deep'] = build_widedeep(input_dim, n_classes)
		models['resnet_mlp'] = build_resnet_mlp(input_dim, n_classes)
		models['conv1d'] = build_conv1d_model(input_dim, n_classes)
		models['se_mlp'] = build_se_mlp(input_dim, n_classes)
		models['bilstm_att'] = build_bilstm_att(input_dim, n_classes)
		models['stacked_gru_ln'] = build_stacked_gru_ln(input_dim, n_classes)
		models['transformer_enc'] = build_transformer_enc(input_dim, n_classes)
		models['mlp_small'] = build_mlp_small(input_dim, n_classes)
		models['mlp_wide'] = build_mlp_wide(input_dim, n_classes)
		models['resnet_conv1d'] = build_resnet_conv1d(input_dim, n_classes)
		return models

	# Build models and print parameter counts
	try:
		models = build_all_models(X.shape[1])
		print('\nBuilt models:')
		for name, m in models.items():
			print(f"  {name}: params={m.count_params()}, input_shape={m.input_shape}, output_shape={m.output_shape}")
		
		# Split data
		print('\nSplitting data...')
		(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_train_val_test(X, y)
		print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
		
		# Diagnostic: check target value distributions
		print('\nTarget value diagnostics:')
		print(f"  Train - unique values: {sorted(set(y_train))}, counts: {pd.Series(y_train).value_counts().sort_index().to_dict()}")
		print(f"  Val   - unique values: {sorted(set(y_val))}, counts: {pd.Series(y_val).value_counts().sort_index().to_dict()}")
		print(f"  Test  - unique values: {sorted(set(y_test))}, counts: {pd.Series(y_test).value_counts().sort_index().to_dict()}")
		print(f"  Input data - min: {X.min():.2f}, max: {X.max():.2f}, mean: {X.mean():.2f}")
		
		# Train all models
		print('\n=== Training Models ===')
		histories = train_all_models(models, X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
		
		# Evaluate on test set
		print('\n=== Test Set Evaluation ===')
		results_dir = Path('results')
		results_dir.mkdir(parents=True, exist_ok=True)
		metrics_lines = []
		for name, model in models.items():
			test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
			print(f"  {name}: test_loss={test_loss:.4f}, test_accuracy={test_acc:.4f}")
			# Predictions and confusion matrix
			pred_probs = model.predict(X_test, verbose=0)
			pred_labels = np.argmax(pred_probs, axis=1)
			cm = confusion_matrix(y_test, pred_labels)
			report = classification_report(y_test, pred_labels, digits=4)
			metrics_lines.append(f"Model: {name}\nTest loss: {test_loss:.4f}\nTest accuracy: {test_acc:.4f}\nConfusion matrix:\n{cm}\nClassification report:\n{report}\n{'-'*60}\n")
		# Save metrics to file
		results_path = results_dir / 'test_results.txt'
		with open(results_path, 'w', encoding='utf-8') as f:
			f.writelines(metrics_lines)
		print(f"\nSaved test metrics and confusion matrices to {results_path}")
		
		print('\nTraining complete! Models saved to models/')
		
	except ImportError as ie:
		print('\nTensorFlow is not available; cannot build models:', ie)
	except Exception as e:
		print('\nError building models:', e)
		import traceback
		traceback.print_exc()


