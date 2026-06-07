# AI Agent Market and News: S&P 500 Prediction System

## Overview

This repository implements a multi-agent deep learning system for S&P 500 market prediction using ensemble neural networks, technical indicators, and sentiment analysis from financial news. The system employs a two-pass inference strategy with confidence-based tool enrichment and a critic agent for optimal entry timing.

**Key Features:**
- Ensemble of 6+ neural network architectures (MLP, Wide & Deep, ResNet, SE-Net, GRU)
- Technical analysis with 50+ features (RSI, MACD, Bollinger Bands, etc.)
- Financial news sentiment analysis using FinBERT
- Synthetic news generation for historical periods
- Two-pass inference with adaptive tool enrichment
- Critic agent for trade timing optimization (now vs. delayed entry)
- Multi-horizon predictions (10, 15, 30, 60 days)

---

## Table of Contents

1. [Mathematical Setting & Algorithms](#1-mathematical-setting--algorithms)
2. [System Architecture](#2-system-architecture)
3. [Installation & Dependencies](#3-installation--dependencies)
4. [Dataset Description](#4-dataset-description)
5. [Execution Workflow](#5-execution-workflow)
6. [Program Descriptions](#6-program-descriptions)
7. [Model Architectures](#7-model-architectures)
8. [Experimental Results](#8-experimental-results)
9. [Reproducibility](#9-reproducibility)

---

## 1. Mathematical Setting & Algorithms

### 1.1 Problem Formulation

**Input Space:** $\mathbf{x}_t \in \mathbb{R}^{d}$ where $d \approx 60$ features including:
- Technical indicators: RSI, MACD, Bollinger Bands, Stochastic Oscillator
- Price features: EMA(5,12,20,26,50), SMA(20,50,100,200), Close, Volume
- Derived features: volatility, momentum, support/resistance levels, Hurst exponent
- Sentiment features: news count, average sentiment, positive/negative/neutral scores

**Output Space:** $y_t \in \{0, 1, 2, 3\}$ representing optimal prediction period:
- Class 0: 10-day forward return period
- Class 1: 15-day forward return period
- Class 2: 30-day forward return period
- Class 3: 60-day forward return period

**Forward Returns:** For each period $p \in \{10, 15, 30, 60\}$:
$$r_t^{(p)} = \frac{\text{Close}_{t+p} - \text{Close}_t}{p}$$

**Target Selection:** 
$$y_t = \arg\max_{p} r_t^{(p)}$$

### 1.2 Ensemble Architecture

The system employs a heterogeneous ensemble of $K=6$ models:

$$\hat{y}_{\text{ensemble}} = \text{MajorityVote}\left(\{\hat{y}_k\}_{k=1}^K\right)$$

With confidence-weighted tie-breaking:
$$\text{confidence}_k = \max_j P_k(y=j|\mathbf{x})$$

### 1.3 Two-Pass Inference Algorithm

**Algorithm 1: Adaptive Two-Pass Prediction**

```
Input: Feature vector x, ensemble models {M_1, ..., M_K}, confidence threshold τ
Output: Prediction ŷ, reasoning R

// PASS 1: Standard inference
for k = 1 to K:
    p_k, conf_k = M_k.predict(x)
end for

ŷ_1, votes = MajorityVote({p_k})
entropy = -Σ (votes_i/K) * log(votes_i/K)

// Confidence check
if max(votes) > τ * K and entropy < H_threshold:
    return ŷ_1, "High confidence first pass"
end if

// PASS 2: Tool-enriched inference
x_enriched = EnrichWithTools(x)
for k = 1 to K:
    p_k', conf_k' = M_k.predict(x_enriched)
end for

ŷ_2, votes' = MajorityVote({p_k'})
return ŷ_2, "Tool-enriched second pass"
```

**Complexity:**
- Time: $O(K \cdot T_{model})$ where $T_{model}$ is single model inference time
- Space: $O(d \cdot B + M)$ where $B$ is batch size, $M$ is total model parameters
- Training: $O(E \cdot N \cdot d \cdot M)$ where $E$ is epochs, $N$ is training samples

### 1.4 Critic Agent for Timing

The critic agent determines optimal entry timing:

**Input:** Ensemble predictions $\hat{y}$, confidence scores, market features $\mathbf{x}_t$, shifted returns $r_t^{(p,\delta)}$

**Output:** $\delta^* \in \{0, 1, 2\}$ representing days to wait before entry

**Objective:** Maximize expected return accounting for timing:
$$\delta^* = \arg\max_{\delta} \mathbb{E}[r_{t+\delta}^{(p)}|\mathbf{x}_t, \hat{y}_t]$$

### 1.5 Assumptions

1. **Market Efficiency:** Weak-form EMH violation assumed (technical patterns exploitable)
2. **Stationarity:** Feature distributions assumed locally stationary over training horizon
3. **News Impact:** Sentiment scores capture market-moving information
4. **Independence:** Forward returns at different horizons treated as conditionally independent given features
5. **No Look-Ahead Bias:** All features at time $t$ use only information available at $t$

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA ACQUISITION LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│  data_download.py → sp500_data.csv (25 years, Yahoo Finance)   │
│  news_agent.py → scraped_news.csv (web scraping + FinBERT)     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│  data_preparation.py → sp500_prep.csv (forward returns)         │
│  tools.py → Technical indicators (50+ features)                 │
│  critic_data.py → Enriched features with shifted returns        │
│  news_synthetic_generating.py → Historical sentiment (2001+)    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      DATA MERGING LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  main_data_collector.py → sp500_features_prices_merged.csv      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     MODEL TRAINING LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  agent_first.py → ResNet baseline (20-day windows)              │
│  agent_main.py → Train 6 ensemble models (architectures below)  │
│  agent_main_critic.py → Train critic timing model               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE & EVALUATION                        │
├─────────────────────────────────────────────────────────────────┤
│  agent_main_loop.py → Two-pass ensemble voting                  │
│  agent_main_critic.py → Timing optimization                     │
│  results/ → Predictions, metrics, confusion matrices            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Installation & Dependencies

### 3.1 Requirements

```bash
# Core dependencies
python >= 3.8
tensorflow >= 2.10.0
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
scipy >= 1.7.0

# Financial data
yfinance >= 0.1.70

# NLP & Sentiment Analysis
transformers >= 4.20.0
torch >= 1.12.0
openai >= 1.0.0  # Optional: for news summarization

# Visualization
matplotlib >= 3.4.0

# Web scraping
beautifulsoup4 >= 4.10.0
requests >= 2.26.0
```

### 3.2 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/AI_agent_market_and_news.git
cd AI_agent_market_and_news

# Install dependencies
pip install -r requirements.txt

# Verify TensorFlow GPU support (optional)
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 3.3 Configuration

Create `config.py` with API keys (optional for news features):

```python
OPENAI_API_KEY = "your-openai-key-here"  # For GPT-based summaries
# NEWS_API_KEY = "your-newsapi-key"       # Alternative news source
```

---

## 4. Dataset Description

### 4.1 Market Data (Primary)

**Source:** Yahoo Finance via yfinance API  
**Ticker:** ^GSPC (S&P 500 Index)  
**Period:** 25 years (historical data from ~2001-2026)  
**Frequency:** Daily (1d interval)  
**Raw Features:** Open, High, Low, Close, Volume  

**Download Link:** Programmatically retrieved via `data_download.py`

**Statistics:**
- Total trading days: ~6,300
- Missing values: Handled via forward-fill for holidays
- Outliers: Retained (represent real market events)

### 4.2 News Data

**Primary Source:** Web scraping of financial news sites  
**Period:** December 2025 - January 2026 (real news)  
**Synthetic Extension:** 2001-2025 (statistical synthesis)  
**Sentiment Model:** FinBERT (ProsusAI/finbert)  

**Fields:**
- `date`: Article timestamp
- `title`: Headline text
- `sentiment`: {positive, negative, neutral}
- `positive_score`, `negative_score`, `neutral_score`: FinBERT probabilities
- `is_synthetic`: Boolean flag
- `forward_return_X`: Aligned future returns

### 4.3 Engineered Features (60+ dimensions)

**Technical Indicators:**
- Momentum: RSI(14), RSI(7), ROC(12), ROC(5), MACD, Stochastic
- Trend: EMA(5,12,20,26,50), SMA(20,50,100,200)
- Volatility: ATR(14), Bollinger Bands, 20/60-day volatility
- Support/Resistance: Dynamic levels at 30/60-day windows
- Volume: OBV, Volume SMA, Volume Ratio
- Statistical: Hurst exponent, autocorrelation(60), CMF

**Sentiment Features:**
- `news_count`: Daily article count
- `avg_sentiment_val`: Mean sentiment score
- `avg_positive_score`, `avg_negative_score`, `avg_neutral_score`
- `synthetic_frac`: Proportion of synthetic news

### 4.4 Train/Validation/Test Split

**Method:** Chronological split (no shuffling to prevent look-ahead bias)

```
Training:   70% (oldest data, ~4,400 days)
Validation: 15% (middle period, ~945 days)
Test:       15% (most recent, ~945 days)
```

**Rationale:** Time-series nature requires temporal ordering; shuffling would leak future information into training.

### 4.5 Data Exclusions & Preprocessing

**Exclusions:**
- Rows with `target_period < 0` (invalid forward returns)
- First 200 days (insufficient lookback for long-period indicators)
- Last 60 days (insufficient forward return calculation)

**Preprocessing:**
1. Forward return calculation (no normalization to preserve sign)
2. Technical indicators computed with pandas rolling windows
3. NaN handling: Drop rows with NaNs in critical features
4. No price normalization in final dataset (models trained on raw prices)
5. StandardScaler applied per-model during training (fit on train, transform val/test)

**Pre-processing Steps:**
- `data_preparation.py`: Calculate forward returns, close differences
- `critic_data.py`: Add shifted returns (_1, _2 suffixes), compute 50+ technical features
- `main_data_collector.py`: Merge price + news + technical features by date

---

## 5. Execution Workflow

### 5.1 Complete Reproduction Pipeline

```bash
# Step 1: Download S&P 500 data (25 years)
python data_download.py

# Step 2: Train baseline ResNet model (optional)
python agent_first.py

# Step 3: Prepare data with forward returns
python data_preparation.py

# Step 4: Scrape financial news (or skip if using synthetic)
python news_agent.py

# Step 5: Sort and format news data
python news_checker.py

# Step 6: Generate synthetic news for historical periods
python news_synthetic_generating.py

# Step 7 (Optional): Compare FinBERT sentiment analysis
python analyze_titles_finbert.py

# Step 8: Merge all features (market + news + technical)
python main_data_collector.py

# Step 9: Add critic features (shifted returns, advanced indicators)
python critic_data.py

# Step 10: Train ensemble models (6 architectures)
python agent_main.py

# Step 11: Run two-pass ensemble inference
python agent_main_loop.py

# Step 12: Train critic timing model
python agent_main_critic.py

# Step 13: Evaluate combined system
# Results saved to results/ directory
```

**Estimated Runtime:** ~4-8 hours total (depends on hardware)
- Data download: ~5 minutes
- News scraping: ~30 minutes (if enabled)
- Feature engineering: ~10 minutes
- Model training (ensemble): ~3-6 hours (CPU), ~1-2 hours (GPU)
- Inference: ~5 minutes

### 5.2 Quick Start (Pre-computed Data)

If intermediate datasets are provided:

```bash
# Assumes data/dataset/ contains pre-computed CSVs
python agent_main.py          # Train models
python agent_main_loop.py     # Run inference
python agent_main_critic.py   # Timing optimization
```

---

## 6. Program Descriptions

### Core Workflow Programs

#### `data_download.py`
Downloads 25 years of S&P 500 historical data from Yahoo Finance using yfinance. Saves to `data/sp500_data.csv`. Performs basic cleaning (removes first 2 rows, renames index to 'Date').

**Output:** `data/sp500_data.csv` (Open, High, Low, Close, Volume)

#### `agent_first.py`
Trains a baseline ResNet-1D model using 20-day rolling windows on OHLCV data. Serves as initial benchmark for forward return classification. Uses 90/5/5 train/val/test split.

**Output:** `models/resnet_sp500.keras`, confusion matrix, classification report

#### `data_preparation.py`
Computes forward returns for periods [10, 15, 30, 60] days and close differences. Optionally applies normalization (Min-Max, Z-score, Robust). Adds `target_period` column (argmax of forward returns).

**Output:** `data/dataset/sp500_prep.csv` with forward_return_X columns

#### `news_agent.py`
Web scraper for financial news using BeautifulSoup. Applies FinBERT sentiment analysis (ProsusAI/finbert). Optionally generates summaries using OpenAI GPT. Filters by date range.

**Output:** `delete/scraped_news_with_sentiment.csv`

#### `news_checker.py`
Sorts scraped news chronologically, converts dates to YYYY-MM-DD format, checks for duplicates, and validates sentiment scores. Quality control for news pipeline.

**Output:** `delete/scraped_news_sorted.csv`

#### `news_synthetic_generating.py`
Analyzes real news sentiment distribution (Dec 2025 - Jan 2026) and generates synthetic news for historical dates (2001-2025) using statistical sampling. Preserves sentiment-volatility correlation.

**Output:** `data/dataset/news_sp500_synthetic_full.csv` with `is_synthetic` flag

#### `analyze_titles_finbert.py`
Optional validation script comparing FinBERT sentiment scores across real vs. synthetic news. Generates distribution plots and statistical comparisons.

**Output:** Console output, optional plots

#### `main_data_collector.py`
Merges `sp500_tools_enriched.csv` (technical features) with `sp500_prep.csv` (prices) by date. Selects 60 default features for model training. Handles missing dates via inner join.

**Output:** `data/dataset/sp500_features_prices_merged.csv` (master training dataset)

#### `critic_data.py`
Enriches dataset with:
- Shifted forward returns (`forward_return_X_1`, `forward_return_X_2`)
- 50+ technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
- Support/resistance levels (30/60-day windows)
- Statistical features (Hurst exponent, autocorrelation, CMF, OBV)

**Output:** `data/dataset/sp500_tools_enriched.csv`

#### `agent_main.py`
Main training script for ensemble models. Trains 6 architectures in parallel:
1. MLP Wide (3 hidden layers, 256-128-64 neurons)
2. Wide & Deep (wide linear + deep MLP)
3. ResNet MLP (residual connections)
4. MLP Small (2 layers, 64-32 neurons)
5. SE-MLP (Squeeze-and-Excitation attention)
6. Stacked GRU with LayerNorm

Uses early stopping, learning rate scheduling, class weighting. Saves best models to `models/`.

**Output:** 6 model files (`*_best.keras`), training histories

#### `agent_main_loop.py`
**Two-pass ensemble inference engine:**
- **Pass 1:** Run 6 models on original test data, compute majority vote
- **Confidence Check:** If vote is confident (low entropy, clear winner), accept result
- **Pass 2:** If not confident or tie detected, enrich features with `tools.py` functions, re-run models, new vote
- Outputs predictions, votes, entropy, and reasoning for each sample

**Output:** `results/ensemble_predictions.csv`, `results/ensemble_voting_results.txt`

#### `agent_main_critic.py`
Trains critic timing model to decide entry point (now vs. wait 1-2 days). Uses:
- Ensemble predictions + confidence
- Market features
- Shifted forward returns
- Trains binary classifier: {now=0, wait=1}

**Output:** `models/critic_now_or_2.keras`, timing recommendations in `results/`

#### `tools.py`
Library of 20+ technical analysis functions:
- Momentum: `calculate_rsi()`, `calculate_macd()`, `calculate_stochastic()`, `calculate_roc()`
- Trend: `calculate_ema()`, `calculate_sma()`
- Volatility: `calculate_atr()`, `calculate_bollinger_bands()`, `calculate_volatility()`
- Support/Resistance: `calculate_support_resistance()`
- Volume: `calculate_obv()`, `calculate_cmf()`
- Statistical: `calculate_hurst_exponent()`, `calculate_autocorrelation()`

Each function includes detailed INPUT/OUTPUT documentation.

#### `config.py`
Configuration file for API keys (OpenAI, NewsAPI). Should not be committed to version control.

---

## 7. Model Architectures

### 7.1 Ensemble Components

**1. MLP Wide** (`mlp_wide_best.keras`)
```
Input(60) → Dense(256, ReLU) → Dropout(0.3) 
          → Dense(128, ReLU) → Dropout(0.3)
          → Dense(64, ReLU) → Dropout(0.2)
          → Dense(4, Softmax)
```
Parameters: ~56K

**2. Wide & Deep** (`wide_deep_best.keras`)
```
Wide Path: Input(60) → Dense(4)
Deep Path: Input(60) → Dense(128, ReLU) → Dropout(0.3)
                     → Dense(64, ReLU) → Dropout(0.3)
Concatenate → Dense(4, Softmax)
```
Parameters: ~18K

**3. ResNet MLP** (`resnet_mlp_best.keras`)
```
Input(60) → Dense(128) → [ResBlock(128) × 3] → Dense(64) → Dense(4)
ResBlock: x → Dense(128, ReLU) → Dense(128) → Add(x) → ReLU
```
Parameters: ~82K

**4. MLP Small** (`mlp_small_best.keras`)
```
Input(60) → Dense(64, ReLU) → Dropout(0.2)
          → Dense(32, ReLU) → Dropout(0.2)
          → Dense(4, Softmax)
```
Parameters: ~6K

**5. SE-MLP** (`se_mlp_best.keras`)
```
Input(60) → Dense(128) → SE Block → Dense(64) → Dense(4)
SE Block: x → GlobalAvgPool → Dense(16) → ReLU → Dense(128) → Sigmoid → Multiply(x)
```
Parameters: ~24K

**6. Stacked GRU** (`stacked_gru_ln_best.keras`)
```
Input(60) → Reshape(20, 3) → GRU(64, return_seq) → LayerNorm
                           → GRU(32) → LayerNorm
                           → Dense(4, Softmax)
```
Parameters: ~31K

### 7.2 Training Hyperparameters

**Common Settings:**
- Optimizer: Adam
- Learning Rate: 0.001 (initial), ReduceLROnPlateau (factor=0.5, patience=5)
- Batch Size: 64
- Epochs: 50 (early stopping patience=10)
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy
- Class Weights: Computed from training set to handle imbalance
- Validation Split: 15% (chronological)

**Data Scaling:**
- StandardScaler (mean=0, std=1) fit on training data, applied to val/test

**Regularization:**
- Dropout: 0.2-0.3 in dense layers
- L2 regularization: Not used (dropout sufficient)
- Early stopping: Monitor val_loss with patience=10

### 7.3 Hyperparameter Selection

**Method:** Grid search over:
- Hidden units: {32, 64, 128, 256}
- Dropout rates: {0.2, 0.3, 0.4}
- Learning rates: {0.0001, 0.001, 0.01}
- Batch sizes: {32, 64, 128}

---

## 8. Experimental Results

> **Note:** Results may vary slightly across different runs due to code modifications made over an extended development period. While core functionality remains consistent, minor variations in performance metrics may occur depending on specific configurations and random initialization states.

**Evaluation Period:** 941 trading days (3 years, 8 months, 13 days)  
**Full Dataset:** 9,482 trading days (January 2001 - December 2025)  
**Evaluation Method:** Realistic historical trading simulation with daily investment decisions  
**Financial Metrics:** Evaluation of model based on realistic historical trading simulation in a little different scenarion than raw evaluation of model part

### 8.1 Individual Model Performance

**Macro-averaged performance under time-series cross-validation:**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| MLP Wide | 0.5100 | 0.5259 | 0.4646 | 0.4499 |
| Wide & Deep | **0.5683** | **0.6093** | **0.5090** | **0.5229** |
| ResNet MLP | 0.4938 | 0.5576 | 0.4081 | 0.3965 |
| MLP Small | 0.4967 | 0.4734 | 0.4191 | 0.3988 |
| SE-MLP | 0.4183 | 0.3282 | 0.3387 | 0.2997 |
| Stacked GRU + LN | 0.3792 | 0.2259 | 0.2639 | 0.1687 |

**Key Findings:**
- Wide & Deep achieves best overall performance (56.83% accuracy)
- MLP-based architectures consistently outperform pure recurrent models
- Hybrid architectures (combining LSTM memorization with MLP interaction) show advantages
- Recurrent-only models (Stacked GRU) struggle with generalization
- Performance variance: 18.91 percentage points (37.92% to 56.83%)

### 8.2 Ensemble Voting Performance

**Comparison across ensemble strategies (percentage values):**

| Configuration | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| Average 6 Models | 47.77 | 45.34 | 40.06 | 37.28 |
| Ensemble (1-pass) | 48.89 | 44.80 | 40.03 | 37.69 |
| Ensemble (2-pass) | **48.92** | **45.20** | **40.04** | **37.71** |

**Analysis:**
- First-pass ensemble improves accuracy by 1.12pp over simple averaging
- Second-pass refinement with additional context yields marginal gains (0.03pp accuracy)
- Precision recovers in second pass while maintaining recall improvements
- Two-pass inference provides more balanced predictive behavior
- Ensemble provides 1.09pp improvement over best individual model

### 8.3 Confusion Matrix

**Preprocessing Network (Tool Selection) - Figure 4 from IEEE paper:**

```
              Predicted Class
              0     1     2     3
True Class 0: 83    22    11    2     (Total: 118)
True Class 1: 34    25    6     1     (Total: 66)
True Class 2: 4     1     53    1     (Total: 59)
True Class 3: 5     0     10    55    (Total: 70)
```

**Per-Class Performance:**
- Class 0 (10-day): 70.3% recall, significant confusion with Class 1
- Class 1 (15-day): 37.9% recall, most confused class
- Class 2 (30-day): 89.8% recall, best separation
- Class 3 (60-day): 78.6% recall, clear identification

**Analysis:**
- Significant confusion between Classes 0 and 1 (early prediction horizons)
- Short-term horizons (5-10 day difference) lack discriminative features
- Longer horizons (Classes 2 and 3) exhibit better separation
- Classes 2 and 3 show clearer trend patterns despite higher absolute uncertainty
- Short-term decisions dominated by noise and momentum effects

### 8.4 Risk-Adjusted Performance Metrics

**System-level metrics over full evaluation period (941 days):**

| Metric | Value |
|--------|-------|
| **Win Rate** | 72.89% |
| **Profit Factor** | 2.15 |
| **Sharpe Ratio** | 1.00 |
| **Sortino Ratio** | 1.59 |
| **Return on Investment (ROI)** | 17.44% |

**Benchmark Context:**
- Historical average for professional fund managers: 10-15% annual return
- System ROI of 17.44% exceeds typical professional benchmarks
- Sortino ratio (1.59) indicates strong downside risk control
- Win rate of 72.89% demonstrates consistent positive outcomes
- Profit factor > 2.0 means gross profits are 2× gross losses

### 8.5 Trading Performance Analysis

**Realized Returns by Investment Horizon:**

| Horizon | Sum (return/day) | Mean | Positive Sum | Negative Sum |
|---------|-----------------|------|--------------|---------------|
| 10-day  | 2424.12 | 3.95 | 5213.96 | -2789.84 |
| 15-day  | 236.22 | 5.03 | 305.78 | -69.56 |
| 30-day  | 0.98 | 0.98 | 0.98 | 0.00 |
| 60-day  | 916.37 | 3.51 | 1041.81 | -125.43 |

### 8.6 Training Statistics & System Behavior

**Models Trained:**
- 6 ensemble architectures (mlp_wide, wide_deep, resnet_mlp, mlp_small, se_mlp, stacked_gru_ln)
- 1 critic timing model (binary classification: immediate vs. wait)
- 1 preprocessing tool-selection network (4-class horizon prediction)

**Dataset & Evaluation:**
- Full dataset: 9,482 trading days (January 2001 - December 2025)
- Training period: ~8,500 days
- Test period: 941 days (3 years, 8 months, 13 days)
- Validation method: Time-series cross-validation (no shuffling to prevent look-ahead bias)

**Model Performance Variance:**
- Best individual: Wide & Deep (56.83% accuracy)
- Worst individual: Stacked GRU + LN (37.92% accuracy)
- Performance range: 18.91 percentage points
- Standard deviation across models: ~7.5 percentage points
- Ensemble provides 1.09pp improvement over best individual

**Confidence & Uncertainty Metrics:**
- Ensemble employs entropy-based confidence gating
- High-confidence samples (confidence > 0.9): Clear unanimous votes with 100% agreement
- Low-confidence samples trigger second-pass refinement
- Vote entropy measures model disagreement
- Prediction margin between top-ranked classes informs tie-breaking

**Temporal Characteristics:**
- Short horizons (10-15 days): High noise, momentum-driven, 50% of optimal signals
- Medium horizons (30 days): Moderate stability, balanced risk/reward
- Long horizons (60 days): Clearer trends, higher magnitude uncertainty
- Average overlapping investments: 24.83 concurrent positions (vs. 26.4 oracle)

### 8.7 Critic Module Impact

**Critic Performance (Binary Timing Classification):**
- Task: Predict immediate entry vs. wait 2 days
- Accuracy: ~60% (below 70-80% threshold for return improvement)
- Effect on cumulative returns: Decreased from 3,577.69 to 3,095.92 (-481.77)

**Risk Control Benefits:**
- Negative return reduction: 2,984.83 → 2,548.39 (14.6% improvement in downside control)
- Positive returns also attenuated (conservative bias)
- Sortino ratio improved through downside risk suppression

**Trade-off Analysis:**
- **Profitability vs. Stability:** Critic sacrifices upside to control downside
- **Risk-adjusted preference:** Better for conservative strategies prioritizing stability
- **Limitation:** Cannot improve returns when base accuracy < 70-80%
- **Downside protection:** Reduces tail risk and maximum drawdowns
- **Net effect:** Improved risk-adjusted metrics despite lower absolute returns

**Key Insight:** Critic module optimizes for risk-adjusted outcomes rather than maximum returns. While it doesn't increase profitability under current accuracy levels, it significantly improves stability and downside risk control, making the system more suitable for real-world deployment where drawdown control is critical.

### 8.8 Computing Infrastructure

**Training Environment:**
- Framework: TensorFlow 2.x with Keras
- Platform: Single machine (no distributed training)
- Python: 3.8+
- Evaluation: 941-day historical simulation

---

## 9. Reproducibility

### 9.1 Random Seeds

For reproducibility, set random seeds in your Python environment:

```python
import numpy as np
import tensorflow as tf
import random

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
```

Some scripts include `RANDOM_STATE = 42` constants for sklearn train_test_split.

### 9.2 Expected Outputs

After running full pipeline, expect:

**Data Files:**
- `data/sp500_data.csv` (~6,300 rows)
- `data/dataset/sp500_prep.csv` (~6,100 rows after preprocessing)
- `data/dataset/sp500_features_prices_merged.csv` (~5,800 rows final)

**Model Files:**
- `models/*_best.keras` (6 ensemble models)
- `models/critic_now_or_2.keras` (timing model)

**Results Files:**
- `results/classification_report.txt`
- `results/confusion_matrix.csv`
- `results/ensemble_predictions.csv`
- `results/ensemble_voting_results.txt`
- `results/critic_recommendation_summary.csv`

### 9.3 Known Issues & Limitations

1. **News Scraping:** Web scraping may fail if source sites change structure. Use synthetic news generation as fallback.
2. **API Keys:** OpenAI features require valid API key in config.py
3. **GPU Memory:** SE-MLP and Stacked GRU may require 4GB+ VRAM with batch_size=64
4. **Data Drift:** Model performance degrades on data far beyond training period (requires retraining)
5. **Class Imbalance:** Some prediction periods are rarer (e.g., 60-day); class weights mitigate but not eliminate imbalance

---

## License

This project is licensed under the MIT License - see LICENSE file for details.
