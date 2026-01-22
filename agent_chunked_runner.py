"""
Run selected agent_main architectures on disjoint chunks of the merged dataset.
- Loads the merged features/prices dataset via agent_main.load_merged_dataset.
- Loads pre-trained models if available; otherwise builds the same architectures.
- Splits the input rows into 6 chronological parts and assigns one part per model.
- Saves per-sample predictions to results/chunked_predictions.csv.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

try:
	import tensorflow as tf
	from tensorflow import keras
	from tensorflow.keras import layers
except Exception:
	tf = None
	keras = None
	layers = None

from agent_main import load_merged_dataset


def _require_tf():
	if tf is None:
		raise ImportError('TensorFlow is required. Install with `pip install tensorflow`.')


# --- Architecture builders (copied from agent_main) ---
def build_widedeep(input_dim: int, n_classes: int = 4):
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


def build_resnet_mlp(input_dim: int, n_classes: int = 4):
	_require_tf()
	inp = layers.Input(shape=(input_dim,))
	x = layers.Dense(128, activation='relu')(inp)
	for _ in range(2):
		y = layers.Dense(128, activation='relu')(x)
		y = layers.Dense(128, activation=None)(y)
		x = layers.Add()([x, y])
		x = layers.Activation('relu')(x)
	x = layers.Dense(64, activation='relu')(x)
	out = layers.Dense(n_classes, activation='softmax')(x)
	m = keras.Model(inputs=inp, outputs=out, name='resnet_mlp')
	m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return m


def build_se_mlp(input_dim: int, n_classes: int = 4, se_ratio: int = 8):
	_require_tf()
	inp = layers.Input(shape=(input_dim,))
	x = layers.Dense(256, activation='relu')(inp)
	x = layers.Dense(128, activation='relu')(x)
	se_units = 128
	se = layers.Dense(max(1, input_dim // se_ratio), activation='relu')(inp)
	se = layers.Dense(se_units, activation='sigmoid')(se)
	x = layers.Multiply()([x, se])
	x = layers.Dense(64, activation='relu')(x)
	out = layers.Dense(n_classes, activation='softmax')(x)
	m = keras.Model(inputs=inp, outputs=out, name='se_mlp')
	m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return m


def build_stacked_gru_ln(input_dim: int, n_classes: int = 4):
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


def build_mlp_small(input_dim: int, n_classes: int = 4):
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


def build_mlp_wide(input_dim: int, n_classes: int = 4):
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


MODEL_BUILDERS: Dict[str, callable] = {
	'mlp_wide': build_mlp_wide,
	'wide_deep': build_widedeep,
	'resnet_mlp': build_resnet_mlp,
	'mlp_small': build_mlp_small,
	'se_mlp': build_se_mlp,
	'stacked_gru_ln': build_stacked_gru_ln,
}


# Files expected to already exist in the models/ directory
MODELS_DIR = Path(__file__).resolve().parent / 'models'
MODEL_FILES: Dict[str, Path] = {
	'mlp_wide': MODELS_DIR / 'mlp_wide_best.keras',
	'wide_deep': MODELS_DIR / 'wide_deep_best.keras',
	'resnet_mlp': MODELS_DIR / 'resnet_mlp_best.keras',
	'mlp_small': MODELS_DIR / 'mlp_small_best.keras',
	'se_mlp': MODELS_DIR / 'se_mlp_best.keras',
	'stacked_gru_ln': MODELS_DIR / 'stacked_gru_ln_best.keras',
}


def load_or_build_models(model_files: Dict[str, Path], input_dim: int, n_classes: int = 4):
	_require_tf()
	MODELS_DIR.mkdir(parents=True, exist_ok=True)
	models: Dict[str, keras.Model] = {}
	for name, path in model_files.items():
		if path.exists():
			print(f'Loading saved model for {name} from {path}')
			models[name] = keras.models.load_model(path, compile=False)
		else:
			print(f'Model file for {name} not found at {path}; building architecture instead')
			builder = MODEL_BUILDERS.get(name)
			if builder is None:
				raise ValueError(f'No builder found for model {name}')
			models[name] = builder(input_dim, n_classes)

		chunked_path = MODELS_DIR / f'chunked_{name}.keras'
		models[name].save(chunked_path)
		print(f'Saved model snapshot to {chunked_path}')
	return models


def split_inputs(X: np.ndarray, parts: int = 6) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
	if parts < 1:
		raise ValueError('parts must be >= 1')
	chunks = np.array_split(X, parts)
	ranges: List[Tuple[int, int]] = []
	start = 0
	for chunk in chunks:
		end = start + len(chunk)
		ranges.append((start, end))
		start = end
	return chunks, ranges


def run_chunked_predictions(
	models: Dict[str, keras.Model],
	chunks: List[np.ndarray],
	ranges: List[Tuple[int, int]],
	y_true: np.ndarray,
):
	predictions = []
	metrics_lines: List[str] = []
	for idx, (name, model) in enumerate(models.items()):
		if idx >= len(chunks):
			raise ValueError('Not enough chunks to give each model unique data. Increase parts.')
		chunk = chunks[idx]
		start, end = ranges[idx]
		chunk_y = y_true[start:end]
		if len(chunk) == 0:
			print(f'Skipping empty chunk {idx + 1} for model {name}')
			continue
		print(f'Predicting with {name} on rows {start}:{end} (chunk {idx + 1})')
		probs = model.predict(chunk, verbose=0)
		labels = np.argmax(probs, axis=1)
		conf = np.max(probs, axis=1)
		predictions.append(pd.DataFrame({
			'model': name,
			'row_index': np.arange(start, end),
			'pred_label': labels,
			'pred_confidence': conf,
		}))

		# Metrics similar to agent_main.py
		correct_probs = probs[np.arange(len(chunk)), chunk_y]
		chunk_loss = float(np.mean(-np.log(np.clip(correct_probs, 1e-7, 1.0))))
		chunk_acc = float(np.mean(labels == chunk_y))
		cm = confusion_matrix(chunk_y, labels)
		report = classification_report(chunk_y, labels, digits=4)
		metrics_lines.append(
			f"Model: {name}\n"
			f"Rows: {start}:{end}\n"
			f"Mean NLL: {chunk_loss:.4f}\n"
			f"Accuracy: {chunk_acc:.4f}\n"
			f"Confusion matrix:\n{cm}\n"
			f"Classification report:\n{report}\n"
			f"{'-'*60}\n"
		)

	return predictions, metrics_lines


def main(parts: int = 6, n_classes: int | None = None):
	X, y, cols, dates = load_merged_dataset()
	input_dim = X.shape[1]
	if n_classes is None:
		n_classes = len(np.unique(y)) if len(y) > 0 else 4
	print(f'Loaded dataset with {len(X)} rows and {input_dim} features; n_classes={n_classes}')

	if parts < len(MODEL_FILES):
		raise ValueError(f'Need at least {len(MODEL_FILES)} parts so each model sees unique data.')

	chunks, ranges = split_inputs(X, parts=parts)
	models = load_or_build_models(MODEL_FILES, input_dim=input_dim, n_classes=n_classes)
	pred_frames, metrics_lines = run_chunked_predictions(models, chunks, ranges, y_true=y)

	results_dir = Path(__file__).resolve().parent / 'results'
	results_dir.mkdir(parents=True, exist_ok=True)
	out_path = results_dir / 'chunked_predictions.csv'
	pd.concat(pred_frames, ignore_index=True).to_csv(out_path, index=False)
	print(f'Saved chunked predictions to {out_path}')

	metrics_path = results_dir / 'chunked_metrics.txt'
	with open(metrics_path, 'w', encoding='utf-8') as f:
		f.writelines(metrics_lines)
	print(f'Saved chunked metrics to {metrics_path}')

	if parts > len(MODEL_FILES):
		unused = parts - len(MODEL_FILES)
		print(f'Note: {unused} chunk(s) remained unused because there are fewer models than parts.')


if __name__ == '__main__':
	main()
