# Collection of technical analysis tools for S&P 500 data
# These work best for different time horizons - notes below each function

import pandas as pd
import numpy as np
from typing import Union, Tuple, List, Dict
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# ============================================================================
# MOMENTUM INDICATORS - work well for short-term (10-15 day) predictions
# ============================================================================

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    # RSI - classic momentum indicator
    # above 70 = overbought, below 30 = oversold
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    # MACD - moving average convergence divergence
    # good for spotting trend changes
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    
    return pd.DataFrame({
        'macd': macd,
        'signal': signal_line,
        'histogram': histogram
    })


def calculate_stochastic(data: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    # stochastic oscillator - compares price to recent range
    # returns %K and %D lines (>80 = overbought, <20 = oversold)
    low_min = data.rolling(window=period).min()
    high_max = data.rolling(window=period).max()
    
    stoch = ((data - low_min) / (high_max - low_min)) * 100
    stoch_k = stoch.rolling(window=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=smooth_d).mean()
    
    return pd.DataFrame({
        'stoch_k': stoch_k,
        'stoch_d': stoch_d
    })


def calculate_rate_of_change(data: pd.Series, period: int = 12) -> pd.Series:
    # ROC - rate of change (momentum indicator)
    # positive = uptrend, negative = downtrend
    roc = ((data - data.shift(period)) / data.shift(period)) * 100
    return roc


# ============================================================================
# TREND INDICATORS - better for medium-term (15-30 day) predictions
# ============================================================================

def calculate_moving_averages(data: pd.Series, periods: List[int] = None) -> pd.DataFrame:
    # SMAs for trend identification
    # golden cross (SMA20 > SMA50) = bullish, death cross = bearish
    if periods is None:
        periods = [20, 50, 100, 200]
    
    result = {}
    for period in periods:
        result[f'sma_{period}'] = data.rolling(window=period).mean()
    
    return pd.DataFrame(result)


def calculate_ema(data: pd.Series, periods: List[int] = None) -> pd.DataFrame:
    # exponential moving averages - weights recent prices more than SMA
    if periods is None:
        periods = [12, 26, 50]
    
    result = {}
    for period in periods:
        result[f'ema_{period}'] = data.ewm(span=period).mean()
    
    return pd.DataFrame(result)


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    # average true range - measures volatility
    # higher ATR = more volatile (useful for stop losses)
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_bb_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    # bollinger bands - volatility-based support/resistance
    # returns upper, middle, lower bands + width + position
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    width = upper - lower
    position = (data - lower) / width
    
    return pd.DataFrame({
        'bb_upper': upper,
        'bb_middle': sma,
        'bb_lower': lower,
        'bb_width': width,
        'bb_position': position
    })


# ============================================================================
# VOLATILITY INDICATORS - BEST FOR RISK ASSESSMENT (ALL TIMEFRAMES)
# ============================================================================

def calculate_historical_volatility(returns: pd.Series, period: int = 20) -> pd.Series:
    # historical volatility - standard deviation of returns
    # annualized by multiplying by sqrt(252)
    volatility = returns.rolling(window=period).std() * np.sqrt(252)
    return volatility


def calculate_parkinson_volatility(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    # parkinson volatility - uses high-low range
    # more responsive to intraday moves
    log_hl = np.log(high / low)
    parkinson = (1 / (4 * np.log(2))) * (log_hl ** 2)
    volatility = parkinson.rolling(window=period).mean() ** 0.5 * np.sqrt(252)
    return volatility


# ============================================================================
# VOLUME INDICATORS - BEST FOR CONFIRMATION (10-30 DAY PREDICTIONS)
# ============================================================================

def calculate_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    # volume SMA - track volume trends
    # when volume > SMA suggests strong directional move
    return volume.rolling(window=period).mean()


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    # on-balance volume - cumulative volume indicator
    # rising OBV = volume supports uptrend, falling = downtrend
    obv = volume.copy()
    obv[close < close.shift()] = -obv
    obv[close == close.shift()] = 0
    return obv.cumsum()


def calculate_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    # chaikin money flow - volume-weighted price momentum
    # >0 = buying pressure, <0 = selling pressure
    mfv = ((close - low) - (high - close)) / (high - low) * volume
    cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    return cmf


# ============================================================================
# PRICE ACTION FEATURES - BEST FOR 30-60 DAY PREDICTIONS
# ============================================================================

def calculate_price_patterns(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    # price action pattern detection
    # looks for higher highs/lows, inside bars, trend strength
    result = pd.DataFrame(index=close.index)
    
    # Higher Highs / Lows
    result['higher_high'] = (high > high.shift(1)).astype(int)
    result['higher_low'] = (low > low.shift(1)).astype(int)
    result['lower_high'] = (high < high.shift(1)).astype(int)
    result['lower_low'] = (low < low.shift(1)).astype(int)
    
    # Inside Bar
    result['inside_bar'] = ((high < high.shift(1)) & (low > low.shift(1))).astype(int)
    
    # Trend Strength
    ma20 = close.rolling(20).mean()
    result['closes_above_ma'] = (close > ma20).rolling(10).sum()
    result['trend_strength'] = result['closes_above_ma'] / 10.0
    
    return result


def calculate_support_resistance(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.DataFrame:
    # support and resistance levels
    # finds key price levels and distance to them
    """
    resistance = high.rolling(window=period).max()
    support = low.rolling(window=period).min()
    
    distance_res = ((resistance - close) / close) * 100
    distance_sup = ((close - support) / close) * 100
    
    result = pd.DataFrame({
        'resistance': resistance,
        'support': support,
        'distance_to_resistance': distance_res,
        'distance_to_support': distance_sup,
        'midpoint': (resistance + support) / 2
    })
    
    return result


# ============================================================================
# CORRELATION & TREND FEATURES - BEST FOR 60+ DAY PREDICTIONS
# ============================================================================

def calculate_correlation_features(returns: pd.Series, period: int = 20) -> pd.DataFrame:
    # auto-correlation and trend detection
    # checks for mean reversion vs momentum persistence
    result = pd.DataFrame(index=returns.index)
    
    # Autocorrelation (detects mean reversion vs momentum)
    result['autocorr'] = returns.rolling(window=period).apply(
        lambda x: x.autocorr() if len(x) > 1 else np.nan, raw=False
    )
    
    # Return trend (slope)
    result['return_trend'] = returns.rolling(window=period).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else np.nan, raw=False
    )
    
    # Volatility
    result['return_volatility'] = returns.rolling(window=period).std()
    
    # Skewness
    result['skewness'] = returns.rolling(window=period).skew()
    
    # Kurtosis
    result['kurtosis'] = returns.rolling(window=period).apply(
        lambda x: stats.kurtosis(x) if len(x) > 3 else np.nan, raw=False
    )
    
    return result


def calculate_hurst_exponent(returns: pd.Series, period: int = 252) -> pd.Series:
    # hurst exponent - detects mean reversion vs trending
    # H < 0.5 = mean reversion, H = 0.5 = random walk, H > 0.5 = trending
    def hurst(ts):
        # calculate hurst exponent
        if len(ts) < 10:
            return np.nan
        lags = range(1, min(100, len(ts)//2))
        tau = []
        for lag in lags:
            tau.append(np.std(np.subtract(ts[lag:], ts[:-lag])))
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return poly[0] * 2
    
    return returns.rolling(window=period).apply(hurst, raw=True)


# ============================================================================
# COMBINED MULTI-HORIZON FEATURE SETS
# ============================================================================

def create_features_10day(df: pd.DataFrame) -> pd.DataFrame:
    # feature set optimized for 10-day predictions
    # momentum indicators, short-term MAs, volume, price action
    features = pd.DataFrame(index=df.index)
    
    # Momentum
    features['rsi_14'] = calculate_rsi(df['Close'], 14)
    features['rsi_7'] = calculate_rsi(df['Close'], 7)
    
    macd = calculate_macd(df['Close'])
    features['macd'] = macd['macd']
    features['macd_signal'] = macd['signal']
    features['macd_histogram'] = macd['histogram']
    
    stoch = calculate_stochastic(df['Close'])
    features['stoch_k'] = stoch['stoch_k']
    features['stoch_d'] = stoch['stoch_d']
    
    features['roc_12'] = calculate_rate_of_change(df['Close'], 12)
    features['roc_5'] = calculate_rate_of_change(df['Close'], 5)
    
    # Trends
    ema = calculate_ema(df['Close'], [5, 12, 20])
    features['ema_5'] = ema['ema_5']
    features['ema_12'] = ema['ema_12']
    features['ema_20'] = ema['ema_20']
    
    # Volatility
    features['atr_14'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
    
    bb = calculate_bb_bands(df['Close'], 20, 2)
    features['bb_position'] = bb['bb_position']
    features['bb_width'] = bb['bb_width']
    
    # Volume
    features['volume_sma'] = calculate_volume_sma(df['Volume'], 20)
    features['volume_ratio'] = df['Volume'] / features['volume_sma']
    
    # Price Action
    patterns = calculate_price_patterns(df['Close'], df['High'], df['Low'])
    features['trend_strength'] = patterns['trend_strength']
    
    return features


def create_features_15day(df: pd.DataFrame) -> pd.DataFrame:
    # feature set optimized for 15-day predictions
    # medium-term momentum, support/resistance, volume
    features = pd.DataFrame(index=df.index)
    
    # Momentum and Trend
    features['rsi_14'] = calculate_rsi(df['Close'], 14)
    
    macd = calculate_macd(df['Close'])
    features['macd_histogram'] = macd['histogram']
    
    # Moving Averages
    ema = calculate_ema(df['Close'], [12, 26, 50])
    features['ema_12'] = ema['ema_12']
    features['ema_26'] = ema['ema_26']
    features['ema_50'] = ema['ema_50']
    
    features['ema_12_26_diff'] = features['ema_12'] - features['ema_26']
    features['ema_26_50_diff'] = features['ema_26'] - features['ema_50']
    
    # Support/Resistance
    sr = calculate_support_resistance(df['High'], df['Low'], df['Close'], 20)
    features['resistance'] = sr['resistance']
    features['support'] = sr['support']
    features['distance_to_resistance'] = sr['distance_to_resistance']
    features['distance_to_support'] = sr['distance_to_support']
    
    # Volatility
    bb = calculate_bb_bands(df['Close'], 20, 2)
    features['bb_position'] = bb['bb_position']
    
    # Volume
    features['obv'] = calculate_obv(df['Close'], df['Volume'])
    features['cmf'] = calculate_cmf(df['High'], df['Low'], df['Close'], df['Volume'], 20)
    
    return features


def create_features_30day(df: pd.DataFrame) -> pd.DataFrame:
    # feature set for 30-day predictions
    # longer-term trends, price patterns, volatility regime
    features = pd.DataFrame(index=df.index)
    
    # Trends
    sma = calculate_moving_averages(df['Close'], [20, 50, 100, 200])
    features['sma_20'] = sma['sma_20']
    features['sma_50'] = sma['sma_50']
    features['sma_100'] = sma['sma_100']
    features['sma_200'] = sma['sma_200']
    
    # Price position relative to MAs
    features['price_sma20_diff'] = (df['Close'] - features['sma_20']) / features['sma_20'] * 100
    features['price_sma50_diff'] = (df['Close'] - features['sma_50']) / features['sma_50'] * 100
    
    # Support/Resistance
    sr = calculate_support_resistance(df['High'], df['Low'], df['Close'], 30)
    features['resistance_30'] = sr['resistance']
    features['support_30'] = sr['support']
    
    # Volatility
    returns = df['Close'].pct_change()
    features['volatility_20'] = calculate_historical_volatility(returns, 20)
    
    bb = calculate_bb_bands(df['Close'], 20, 2)
    features['bb_width'] = bb['bb_width']
    
    # Volume trend
    features['volume_trend'] = df['Volume'].rolling(20).mean()
    
    # Price patterns
    patterns = calculate_price_patterns(df['Close'], df['High'], df['Low'])
    features['trend_strength'] = patterns['trend_strength']
    
    return features


def create_features_60day(df: pd.DataFrame) -> pd.DataFrame:
    # feature set for 60-day predictions
    # long-term trends, market regime, mean reversion
    features = pd.DataFrame(index=df.index)
    
    # Long-term Trends
    sma = calculate_moving_averages(df['Close'], [50, 100, 200])
    features['sma_50'] = sma['sma_50']
    features['sma_100'] = sma['sma_100']
    features['sma_200'] = sma['sma_200']
    
    # Trend identification
    features['uptrend'] = (features['sma_50'] > features['sma_100']).astype(int)
    features['strong_uptrend'] = (features['sma_100'] > features['sma_200']).astype(int)
    
    # Support/Resistance - Longer term
    sr = calculate_support_resistance(df['High'], df['Low'], df['Close'], 60)
    features['resistance_60'] = sr['resistance']
    features['support_60'] = sr['support']
    
    # Volatility Regime
    returns = df['Close'].pct_change()
    features['volatility_60'] = calculate_historical_volatility(returns, 60)
    
    # Mean Reversion vs Momentum
    features['autocorr_60'] = calculate_correlation_features(returns, 60)['autocorr']
    features['hurst_exponent'] = calculate_hurst_exponent(returns, 252)
    
    # Price momentum
    features['return_60d'] = (df['Close'] - df['Close'].shift(60)) / df['Close'].shift(60) * 100
    features['return_30d'] = (df['Close'] - df['Close'].shift(30)) / df['Close'].shift(30) * 100
    
    # Volume trend
    features['volume_avg_60'] = df['Volume'].rolling(60).mean()
    
    return features


def create_default_features(df: pd.DataFrame) -> pd.DataFrame:
    # comprehensive feature set for all timeframes
    # includes momentum, MAs, volatility, support/resistance, volume
    features = pd.DataFrame(index=df.index)
    
    # ===== CORE MOMENTUM (Used in all prediction models) =====
    features['rsi_14'] = calculate_rsi(df['Close'], 14)
    
    macd = calculate_macd(df['Close'])
    features['macd'] = macd['macd']
    features['macd_histogram'] = macd['histogram']
    
    # ===== MOVING AVERAGES (Trend identification) =====
    ema = calculate_ema(df['Close'], [12, 26, 50])
    features['ema_12'] = ema['ema_12']
    features['ema_26'] = ema['ema_26']
    features['ema_50'] = ema['ema_50']
    
    sma = calculate_moving_averages(df['Close'], [20, 50, 200])
    features['sma_20'] = sma['sma_20']
    features['sma_50'] = sma['sma_50']
    features['sma_200'] = sma['sma_200']
    
    # ===== VOLATILITY (Risk assessment) =====
    features['atr_14'] = calculate_atr(df['High'], df['Low'], df['Close'], 14)
    
    returns = df['Close'].pct_change()
    features['volatility_20'] = calculate_historical_volatility(returns, 20)
    
    bb = calculate_bb_bands(df['Close'], 20, 2)
    features['bb_position'] = bb['bb_position']
    features['bb_width'] = bb['bb_width']
    
    # ===== SUPPORT/RESISTANCE (Key levels) =====
    sr = calculate_support_resistance(df['High'], df['Low'], df['Close'], 20)
    features['resistance'] = sr['resistance']
    features['support'] = sr['support']
    features['midpoint'] = sr['midpoint']
    
    # ===== VOLUME (Confirmation) =====
    features['volume_sma'] = calculate_volume_sma(df['Volume'], 20)
    features['volume_ratio'] = df['Volume'] / features['volume_sma']
    features['cmf'] = calculate_cmf(df['High'], df['Low'], df['Close'], df['Volume'], 20)
    
    # ===== PRICE ACTION =====
    features['price_return_5d'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
    features['price_return_10d'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
    features['price_return_20d'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20) * 100
    
    # ===== TREND STRENGTH =====
    patterns = calculate_price_patterns(df['Close'], df['High'], df['Low'])
    features['trend_strength'] = patterns['trend_strength']
    
    return features


# ============================================================================
# FEATURE NORMALIZATION & SCALING
# ============================================================================

def scale_features(features: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, Dict]:
    # scale features to standardized ranges
    # method: 'standard' (mean=0, std=1), 'minmax' (0-1), or 'robust'
    features_clean = features.fillna(features.mean())
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
    
    scaled = scaler.fit_transform(features_clean)
    scaled_df = pd.DataFrame(scaled, columns=features_clean.columns, index=features_clean.index)
    
    return scaled_df, scaler


# ============================================================================
# FEATURE ANALYSIS & DIAGNOSTICS
# ============================================================================

def analyze_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    # analyze correlation between features and target
    # returns features ranked by absolute correlation with p-values
    correlations = []
    
    for col in X.columns:
        if X[col].notna().sum() > 10:  # Need at least 10 observations
            mask = X[col].notna() & y.notna()
            if mask.sum() > 10:
                corr, pval = stats.pearsonr(X.loc[mask, col], y[mask])
                correlations.append({
                    'feature': col,
                    'correlation': corr,
                    'p_value': pval,
                    'abs_correlation': abs(corr)
                })
    
    result = pd.DataFrame(correlations)
    result = result.sort_values('abs_correlation', ascending=False)
    
    return result


def remove_collinear_features(features: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    # remove highly correlated features to reduce multicollinearity
    # threshold: correlation above this gets removed (default 0.95)
    """
    corr_matrix = features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Removing {len(to_drop)} collinear features: {to_drop}")
    return features.drop(columns=to_drop)


if __name__ == "__main__":
    print("Analytics Tools Module Loaded")
    print("Use: from tools import *")
