"""
Run inference with chunked_* snapshots of the agent_main architectures and save predictions.
- Loads the merged dataset via agent_main.load_merged_dataset.
- Loads chunked model weights from models/chunked_{name}.keras.
- Produces per-model predictions and an averaged-probability ensemble.
- Saves CSVs and a brief text report in results/.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
	import tensorflow as tf
	from tensorflow import keras
except Exception:
	tf = None
	keras = None

from agent_main import load_merged_dataset

MODELS_DIR = Path(__file__).resolve().parent / 'models'
RESULTS_DIR = Path(__file__).resolve().parent / 'results'

CHUNKED_MODEL_PATHS: Dict[str, Path] = {
	'mlp_wide': MODELS_DIR / 'chunked_mlp_wide.keras',
	'wide_deep': MODELS_DIR / 'chunked_wide_deep.keras',
	'resnet_mlp': MODELS_DIR / 'chunked_resnet_mlp.keras',
	'mlp_small': MODELS_DIR / 'chunked_mlp_small.keras',
	'se_mlp': MODELS_DIR / 'chunked_se_mlp.keras',
	'stacked_gru_ln': MODELS_DIR / 'chunked_stacked_gru_ln.keras',
}


def _require_tf():
	if tf is None:
		raise ImportError('TensorFlow is required. Install with `pip install tensorflow`.')


def load_chunked_models() -> Dict[str, keras.Model]:
	_require_tf()
	missing: List[str] = []
	models: Dict[str, keras.Model] = {}
	for name, path in CHUNKED_MODEL_PATHS.items():
		if not path.exists():
			missing.append(str(path))
			continue
		print(f'Loading {name} from {path}')
		models[name] = keras.models.load_model(path, compile=False)
	if missing:
		raise FileNotFoundError(f"Missing chunked model files: {missing}")
	return models


def predict_models(models: Dict[str, keras.Model], X: np.ndarray, dates: np.ndarray | List[object]) -> Tuple[List[pd.DataFrame], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
	per_model_frames: List[pd.DataFrame] = []
	all_probs: List[np.ndarray] = []
	probs_dict: Dict[str, np.ndarray] = {}
	vote_matrix: List[np.ndarray] = []
	for name, model in models.items():
		probs = model.predict(X, verbose=0)
		labels = np.argmax(probs, axis=1)
		conf = np.max(probs, axis=1)
		per_model_frames.append(pd.DataFrame({
			'model': name,
			'date': dates,
			'pred_label': labels,
			'pred_confidence': conf,
		}))
		all_probs.append(probs)
		probs_dict[name] = probs
		vote_matrix.append(labels)
	ensemble_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
	votes = np.stack(vote_matrix, axis=0) if vote_matrix else np.empty((0, len(X)))
	return per_model_frames, ensemble_probs, probs_dict, votes


def build_ensemble_frame(ensemble_probs: np.ndarray, votes: np.ndarray, dates: np.ndarray | List[object]) -> pd.DataFrame:
	labels = np.argmax(ensemble_probs, axis=1)
	conf = np.max(ensemble_probs, axis=1)
	# majority vote; tie-break with probability ensemble
	if votes.size > 0:
		vote_majority = []
		for i in range(votes.shape[1]):
			counts = np.bincount(votes[:, i].astype(int))
			max_vote = counts.max()
			cands = np.flatnonzero(counts == max_vote)
			if len(cands) == 1:
				vote_majority.append(cands[0])
			else:
				vote_majority.append(labels[i])
		vote_majority = np.array(vote_majority, dtype=int)
	else:
		vote_majority = labels

	data = {
		'date': dates,
		'ensemble_pred': labels,
		'ensemble_vote_pred': vote_majority,
		'ensemble_confidence': conf,
	}
	for cls in range(ensemble_probs.shape[1]):
		data[f'prob_class_{cls}'] = ensemble_probs[:, cls]
	return pd.DataFrame(data)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, object]:
	preds = np.argmax(probs, axis=1)
	acc = accuracy_score(y_true, preds)
	report = classification_report(y_true, preds, digits=4)
	cm = confusion_matrix(y_true, preds)
	return {'accuracy': acc, 'report': report, 'cm': cm}


def compute_vote_metrics(y_true: np.ndarray, votes: np.ndarray) -> Dict[str, object]:
	acc = accuracy_score(y_true, votes)
	report = classification_report(y_true, votes, digits=4)
	cm = confusion_matrix(y_true, votes)
	return {'accuracy': acc, 'report': report, 'cm': cm}


def save_outputs(per_model_frames: List[pd.DataFrame], ensemble_df: pd.DataFrame, votes: np.ndarray, model_order: List[str], dates: np.ndarray | List[object]):
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)

	long_path = RESULTS_DIR / 'chunked_main_predictions.csv'
	pd.concat(per_model_frames, ignore_index=True).to_csv(long_path, index=False)
	print(f'Saved per-model predictions to {long_path}')

	ens_path = RESULTS_DIR / 'chunked_main_ensemble.csv'
	ensemble_df.to_csv(ens_path, index=False)
	print(f'Saved ensemble predictions to {ens_path}')

	# Save raw voting outcomes per model for each row
	if votes.size > 0:
		vote_df = pd.DataFrame({'date': dates})
		for idx, name in enumerate(model_order):
			vote_df[f'{name}_vote'] = votes[idx]
		vote_df['ensemble_vote_pred'] = ensemble_df['ensemble_vote_pred'].values
		vote_path = RESULTS_DIR / 'chunked_main_votes.csv'
		vote_df.to_csv(vote_path, index=False)
		print(f'Saved vote breakdown to {vote_path}')


def write_report(
	models: Dict[str, keras.Model],
	X: np.ndarray,
	y: np.ndarray,
	ensemble_df: pd.DataFrame,
	ensemble_probs: np.ndarray,
	probs_dict: Dict[str, np.ndarray],
):
	report_path = RESULTS_DIR / 'chunked_main_report.txt'
	with open(report_path, 'w', encoding='utf-8') as f:
		f.write('Chunked model inference report\n')
		f.write('='*60 + '\n')
		f.write(f'Models used: {list(models.keys())}\n')
		f.write(f'Samples: {len(X)}\n')
		f.write(f'Input dim: {X.shape[1]}\n')
		f.write(f'Ensemble predicted classes: {sorted(ensemble_df["ensemble_pred"].unique())}\n')
		f.write(f'Ensemble confidence mean: {ensemble_df["ensemble_confidence"].mean():.4f}\n')
		f.write(f'Ensemble confidence min: {ensemble_df["ensemble_confidence"].min():.4f}\n')
		f.write(f'Ensemble confidence max: {ensemble_df["ensemble_confidence"].max():.4f}\n')

		f.write('\nEnsemble prediction distribution (overall voting)\n')
		f.write('-'*60 + '\n')
		total = len(ensemble_df)
		for cls, count in sorted(ensemble_df['ensemble_pred'].value_counts().items()):
			pct = count / total * 100 if total else 0
			f.write(f'Class {cls}: {count} ({pct:5.1f}%)\n')

		f.write('\nActual label distribution\n')
		f.write('-'*60 + '\n')
		total_y = len(y)
		for cls, count in sorted(pd.Series(y).value_counts().items()):
			pct = count / total_y * 100 if total_y else 0
			f.write(f'Class {cls}: {count} ({pct:5.1f}%)\n')

		f.write('\n=== Metrics: Ensemble (probability-averaged) ===\n')
		ens_metrics = compute_metrics(y, ensemble_probs)
		f.write(f"Accuracy: {ens_metrics['accuracy']:.4f}\n")
		f.write('Confusion matrix:\n')
		f.write(str(ens_metrics['cm']) + '\n')
		f.write('Classification report:\n')
		f.write(ens_metrics['report'] + '\n')

		f.write('\n=== Metrics: Ensemble (majority vote with prob tie-break) ===\n')
		vote_metrics = compute_vote_metrics(y, ensemble_df['ensemble_vote_pred'].values)
		f.write(f"Accuracy: {vote_metrics['accuracy']:.4f}\n")
		f.write('Confusion matrix:\n')
		f.write(str(vote_metrics['cm']) + '\n')
		f.write('Classification report:\n')
		f.write(vote_metrics['report'] + '\n')

		f.write('\n=== Metrics: Per-model ===\n')
		for name, probs in probs_dict.items():
			m = compute_metrics(y, probs)
			f.write(f'Model: {name}\n')
			f.write(f"  Accuracy: {m['accuracy']:.4f}\n")
			f.write(f"  Confusion matrix:\n{m['cm']}\n")
			f.write('  Classification report:\n')
			f.write(m['report'] + '\n')
			f.write('-'*60 + '\n')
	print(f'Saved report to {report_path}')


def main():
	X, y, cols, dates = load_merged_dataset()
	print(f'Loaded merged dataset: rows={len(X)}, features={X.shape[1]}')
	models = load_chunked_models()
	per_model_frames, ensemble_probs, probs_dict, votes = predict_models(models, X, dates)
	ensemble_df = build_ensemble_frame(ensemble_probs, votes, dates)
	save_outputs(per_model_frames, ensemble_df, votes, list(models.keys()), dates)
	write_report(models, X, y, ensemble_df, ensemble_probs, probs_dict)


if __name__ == '__main__':
	main()
