# Two-Pass Ensemble Inference Implementation

## Overview
Successfully implemented the two-pass conditional ensemble voting system in `agent_main_loop.py`.

## Architecture

### PASS 1: Original Features (67 dimensions)
- Load merged dataset: `sp500_features_prices_merged.csv`
- Split data chronologically: 70% train, 12.75% val, 17.25% test
- Run 6 models on original test features:
  - mlp_wide_best.keras
  - wide_deep_best.keras
  - resnet_mlp_best.keras
  - mlp_small_best.keras
  - se_mlp_best.keras
  - stacked_gru_ln_best.keras
- Apply ensemble voting with majority vote + confidence-based tie-breaking
- Compute confidence metrics and reasoning

### Confidence Check Decision Logic
After Pass 1, evaluate confidence with thresholds:
- **Confidence Threshold**: mean_confidence > 0.6
- **Entropy Threshold**: mean_entropy < 1.0 (lower = more agreement)
- **Majority Threshold**: mean_majority_percentage > 50%
- **Tie Check**: tie_resolved_count == 0

**Condition for Pass 2**:
- If `NOT (all_thresholds_met AND no_ties)` → Run Pass 2
- Otherwise → Use Pass 1 results only

### PASS 2: Tool-Enriched Features (117 dimensions)
Only executed if Pass 1 fails confidence check or has ties:

1. **Tool Feature Generation**:
   - Extract OHLCV columns from full dataset
   - Call `create_default_features()` from `tools.py` (generates 40+ technical indicators)
   - Extract test portion matching test set indices
   - Fill NaN/inf values with 0
   - Apply StandardScaler normalization (mean=0, std=1)

2. **Feature Concatenation**:
   - Concatenate original 67 features + 50 tool features = 117 total

3. **Model Inference**:
   - Run same 6 models on enriched X_test_enriched
   - Apply ensemble voting
   - Compute new confidence metrics and reasoning

4. **Results Comparison**:
   - Compare Pass 1 vs Pass 2 metrics
   - Report improvements in confidence, entropy, majority %, and tie resolution

## Output Files

### `results/ensemble_voting_results.txt`
Comprehensive report including:
- **Section 1**: Pass 1 Results
  - Input dimensions
  - Confidence metrics (mean, std dev)
  - Entropy stats
  - Tie and low-confidence sample counts

- **Section 2**: Confidence Check Results
  - Threshold evaluations (yes/no for each)
  - Pass 1 confidence status
  - Pass 2 execution status

- **Section 3**: Pass 2 Results (if executed)
  - Tool features information
  - Enriched input dimensions
  - New confidence metrics

- **Section 4**: Comparison (if Pass 2 executed)
  - Confidence improvement
  - Entropy change
  - Majority % improvement
  - Tie resolution count difference

- **Section 5**: Classification Metrics
  - Accuracy, Precision, Recall, F1 (macro & weighted)
  - Confusion matrix
  - Classification report

- **Section 6**: Prediction Distribution
  - Count and percentage for each class (0-3)

- **Section 7**: Per-Sample Breakdown (5 random samples)
  - True label and ensemble prediction
  - Confidence, entropy, majority %
  - Vote counts and aggregate probabilities
  - Per-model predictions

### `results/ensemble_predictions.csv`
CSV with all samples:
- sample_id
- ensemble_prediction
- ensemble_confidence
- vote_entropy
- majority_percentage
- true_label
- p0, p1, p2, p3 (class probabilities)

## Key Functions

### `load_merged_dataset()`
Loads normalized feature dataset and returns X, y, input_cols, dates.

### `load_best_models()`
Loads 6 .keras models from models/ directory.

### `ensemble_vote(all_predictions, all_probabilities)`
Performs majority voting with aggregate probability tie-breaking.
Returns: vote_counts, predictions, confidences, entropies, majority_percentages, agg_probs, per_model_confidences, n_models, n_samples.

### `calculate_confidence_metrics(vote_results)`
Computes mean/std/min/max confidence, entropy stats, agreement distribution, prediction counts.

### `compute_reasoning(vote_results, conf_thresh=0.6, entropy_thresh=1.0, majority_thresh=50.0)`
Detects ties and low-confidence cases, computes margins, returns per-sample reasoning dict.

### `generate_tool_features(df_full, test_start_idx, test_size)`
Generates tool features from OHLCV, normalizes with StandardScaler, returns n_test × n_tools array.

## Execution Flow

```
main()
  ├─ Load data and split (70/12.75/17.25)
  ├─ Load 6 models
  ├─ PASS 1:
  │  ├─ Batch inference on X_test (67 dims)
  │  ├─ Ensemble voting
  │  ├─ Confidence metrics + reasoning
  │  └─ Print Pass 1 summary
  ├─ CONFIDENCE CHECK:
  │  ├─ Evaluate: conf > 0.6 AND entropy < 1.0 AND majority > 50%
  │  └─ Check: no ties?
  ├─ IF confident AND no ties:
  │  └─ Use Pass 1 results (pass2_executed=False)
  ├─ ELSE:
  │  ├─ PASS 2:
  │  │  ├─ Load full dataset
  │  │  ├─ Generate tool features (StandardScaler normalized)
  │  │  ├─ Concatenate original + tools → 117 dims
  │  │  ├─ Batch inference on X_test_enriched
  │  │  ├─ Ensemble voting on enriched
  │  │  ├─ New confidence metrics + reasoning
  │  │  └─ Print Pass 2 summary & comparison
  │  └─ Use Pass 2 results (pass2_executed=True)
  ├─ SAVE RESULTS:
  │  ├─ Compute classification metrics (accuracy, precision, recall, F1, confusion matrix)
  │  ├─ Write comprehensive txt report with Pass 1/2 comparison
  │  ├─ Save detailed CSV predictions
  │  └─ Print execution summary
  └─ Exit
```

## Thresholds and Parameters

- **Confidence Threshold**: 0.6 (60% max probability)
- **Entropy Threshold**: 1.0 (entropy of vote agreement)
- **Majority Threshold**: 50% (% of models voting for winner)
- **Train/Val/Test Split**: 70% / 12.75% / 17.25% (seed=42 for reproducibility)
- **StandardScaler**: Applied to both original merged features and tool features

