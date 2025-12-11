"""
Analytical Tools for S&P 500 Market Data Analysis and Prediction

This module provides comprehensive tools for analyzing sp500_prep.csv data
and creating features optimized for prediction at different time horizons
(10, 15, 30, and 60 days ahead).

Each tool includes:
- INPUT: Description of expected input data format
- OUTPUT: Description of returned data format and columns
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple, List, Dict
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# ============================================================================
# MOMENTUM INDICATORS - BEST FOR 10-15 DAY PREDICTIONS
# ============================================================================

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) - Momentum oscillator
    
    INPUT:
        data: pd.Series of closing prices
        period: int, lookback period (default 14)
    
    OUTPUT:
        pd.Series: RSI values (0-100), where >70 = overbought, <30 = oversold
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence) - Trend-following momentum indicator
    
    INPUT:
        data: pd.Series of closing prices
        fast: int, fast EMA period (default 12)
        slow: int, slow EMA period (default 26)
        signal: int, signal line period (default 9)
    
    OUTPUT:
        pd.DataFrame with columns:
            - macd: MACD line (fast - slow EMA)
            - signal: Signal line (EMA of MACD)
            - histogram: MACD histogram (MACD - Signal)
    """
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
    """
    Stochastic Oscillator - Momentum indicator comparing price to price range
    
    INPUT:
        data: pd.Series of closing prices
        period: int, lookback period (default 14)
        smooth_k: int, K line smoothing (default 3)
        smooth_d: int, D line smoothing (default 3)
    
    OUTPUT:
        pd.DataFrame with columns:
            - stoch_k: %K line (0-100)
            - stoch_d: %D line (SMA of %K)
            Values: >80 = overbought, <20 = oversold
    """
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
    """
    Rate of Change (ROC) - Measures momentum and price velocity
    
    INPUT:
        data: pd.Series of closing prices
        period: int, lookback period (default 12)
    
    OUTPUT:
        pd.Series: ROC percentage changes
            Positive = uptrend, Negative = downtrend
    """
    roc = ((data - data.shift(period)) / data.shift(period)) * 100
    return roc


# ============================================================================
# TREND INDICATORS - BEST FOR 15-30 DAY PREDICTIONS
# ============================================================================

def calculate_moving_averages(data: pd.Series, periods: List[int] = None) -> pd.DataFrame:
    """
    Multiple Moving Averages (SMA) - Trend identification and support/resistance
    
    INPUT:
        data: pd.Series of closing prices
        periods: List[int], periods to calculate (default [20, 50, 100, 200])
    
    OUTPUT:
        pd.DataFrame with columns: sma_20, sma_50, sma_100, sma_200
            - Golden Cross (SMA20 > SMA50) = bullish signal
            - Death Cross (SMA20 < SMA50) = bearish signal
    """
    if periods is None:
        periods = [20, 50, 100, 200]
    
    result = {}
    for period in periods:
        result[f'sma_{period}'] = data.rolling(window=period).mean()
    
    return pd.DataFrame(result)


def calculate_ema(data: pd.Series, periods: List[int] = None) -> pd.DataFrame:
    """
    Exponential Moving Averages (EMA) - Faster trend response than SMA
    
    INPUT:
        data: pd.Series of closing prices
        periods: List[int], periods to calculate (default [12, 26, 50])
    
    OUTPUT:
        pd.DataFrame with columns: ema_12, ema_26, ema_50
            EMA gives more weight to recent prices for quicker trend detection
    """
    if periods is None:
        periods = [12, 26, 50]
    
    result = {}
    for period in periods:
        result[f'ema_{period}'] = data.ewm(span=period).mean()
    
    return pd.DataFrame(result)


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR) - Volatility indicator
    
    INPUT:
        high: pd.Series of high prices
        low: pd.Series of low prices
        close: pd.Series of closing prices
        period: int, lookback period (default 14)
    
    OUTPUT:
        pd.Series: ATR values showing volatility magnitude
            Higher ATR = higher volatility, useful for stop-loss placement
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_bb_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    """
    Bollinger Bands - Volatility-based support/resistance
    
    INPUT:
        data: pd.Series of closing prices
        period: int, SMA period (default 20)
        std_dev: int, standard deviations (default 2)
    
    OUTPUT:
        pd.DataFrame with columns:
            - bb_upper: Upper band (SMA + 2*StdDev)
            - bb_middle: Middle band (SMA)
            - bb_lower: Lower band (SMA - 2*StdDev)
            - bb_width: Band width (volatility measure)
            - bb_position: % position between bands (0-1)
    """
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
    """
    Historical Volatility - Standard deviation of returns
    
    INPUT:
        returns: pd.Series of daily returns (percentage or decimal)
        period: int, lookback period (default 20)
    
    OUTPUT:
        pd.Series: Annualized volatility (if daily returns, multiply by sqrt(252))
            Used to assess market risk and option pricing
    """
    volatility = returns.rolling(window=period).std() * np.sqrt(252)
    return volatility


def calculate_parkinson_volatility(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """
    Parkinson Volatility - Uses high-low range for better volatility estimate
    
    INPUT:
        high: pd.Series of high prices
        low: pd.Series of low prices
        period: int, lookback period (default 20)
    
    OUTPUT:
        pd.Series: Parkinson volatility measure
            More responsive to intraday price movements than historical volatility
    """
    log_hl = np.log(high / low)
    parkinson = (1 / (4 * np.log(2))) * (log_hl ** 2)
    volatility = parkinson.rolling(window=period).mean() ** 0.5 * np.sqrt(252)
    return volatility


# ============================================================================
# VOLUME INDICATORS - BEST FOR CONFIRMATION (10-30 DAY PREDICTIONS)
# ============================================================================

def calculate_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Volume Simple Moving Average - Track volume trends
    
    INPUT:
        volume: pd.Series of trading volumes
        period: int, lookback period (default 20)
    
    OUTPUT:
        pd.Series: Average volume over period
            Volume > SMA suggests strong directional move
    """
    return volume.rolling(window=period).mean()


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume (OBV) - Cumulative volume indicator
    
    INPUT:
        close: pd.Series of closing prices
        volume: pd.Series of volumes
    
    OUTPUT:
        pd.Series: Cumulative OBV values
            Rising OBV = volume supports uptrend
            Falling OBV = volume supports downtrend
    """
    obv = volume.copy()
    obv[close < close.shift()] = -obv
    obv[close == close.shift()] = 0
    return obv.cumsum()


def calculate_cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Chaikin Money Flow (CMF) - Volume-weighted price momentum
    
    INPUT:
        high: pd.Series of high prices
        low: pd.Series of low prices
        close: pd.Series of closing prices
        volume: pd.Series of volumes
        period: int, lookback period (default 20)
    
    OUTPUT:
        pd.Series: CMF values (-1 to 1)
            > 0 = buying pressure, < 0 = selling pressure
    """
    mfv = ((close - low) - (high - close)) / (high - low) * volume
    cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    return cmf


# ============================================================================
# PRICE ACTION FEATURES - BEST FOR 30-60 DAY PREDICTIONS
# ============================================================================

def calculate_price_patterns(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """
    Price Action Pattern Features - Detect reversal and continuation patterns
    
    INPUT:
        close: pd.Series of closing prices
        high: pd.Series of high prices
        low: pd.Series of low prices
    
    OUTPUT:
        pd.DataFrame with columns:
            - higher_high: 1 if new higher high, 0 otherwise
            - higher_low: 1 if new higher low, 0 otherwise
            - lower_high: 1 if new lower high, 0 otherwise
            - lower_low: 1 if new lower low, 0 otherwise
            - inside_bar: 1 if inside bar (both high and low within previous range)
            - trend_strength: Ratio of closes above/below MA
    """
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
    """
    Support and Resistance Levels - Key price levels
    
    INPUT:
        high: pd.Series of high prices
        low: pd.Series of low prices
        close: pd.Series of closing prices
        period: int, lookback period (default 20)
    
    OUTPUT:
        pd.DataFrame with columns:
            - resistance: Highest high in period
            - support: Lowest low in period
            - distance_to_resistance: Percentage distance to resistance
            - distance_to_support: Percentage distance to support
            - level_strength: Number of times price touched level
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
    """
    Auto-correlation and Trend Features - Detect persistence and mean reversion
    
    INPUT:
        returns: pd.Series of daily returns
        period: int, lookback period (default 20)
    
    OUTPUT:
        pd.DataFrame with columns:
            - autocorr: Autocorrelation of returns (mean reversion vs momentum)
            - return_trend: Linear trend of returns
            - return_volatility: Volatility of returns
            - skewness: Distribution skewness
    """
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
    """
    Hurst Exponent - Detects mean reversion vs trending behavior
    
    INPUT:
        returns: pd.Series of daily returns
        period: int, lookback period (default 252, ~1 year)
    
    OUTPUT:
        pd.Series: Hurst exponent values
            H < 0.5: Mean reversion (price tends to reverse)
            H = 0.5: Random walk
            H > 0.5: Trending (momentum persists)
    """
    def hurst(ts):
        """Calculate Hurst Exponent"""
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
    """
    Create feature set optimized for 10-day price prediction
    
    INPUT:
        df: pd.DataFrame with columns: Close, High, Low, Volume
    
    OUTPUT:
        pd.DataFrame with engineered features:
            - Momentum indicators (RSI, Stochastic, ROC)
            - Short-term moving averages
            - Volume indicators
            - Price action patterns
    """
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
    """
    Create feature set optimized for 15-day price prediction
    
    INPUT:
        df: pd.DataFrame with columns: Close, High, Low, Volume
    
    OUTPUT:
        pd.DataFrame with engineered features:
            - Medium-term momentum and trend
            - Support/resistance levels
            - Volume confirmation
            - Volatility patterns
    """
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
    """
    Create feature set optimized for 30-day price prediction
    
    INPUT:
        df: pd.DataFrame with columns: Close, High, Low, Volume
    
    OUTPUT:
        pd.DataFrame with engineered features:
            - Medium to long-term trends
            - Price patterns and levels
            - Volatility regime
            - Volume trends
    """
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
    """
    Create feature set optimized for 60-day price prediction
    
    INPUT:
        df: pd.DataFrame with columns: Close, High, Low, Volume
    
    OUTPUT:
        pd.DataFrame with engineered features:
            - Long-term trend identification
            - Market regime analysis
            - Structural support/resistance
            - Mean reversion indicators
    """
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
    """
    Create comprehensive default feature set - Most useful across all timeframes
    
    INPUT:
        df: pd.DataFrame with columns: Close, High, Low, Volume
    
    OUTPUT:
        pd.DataFrame with engineered features including:
            - Core momentum indicators (RSI, MACD)
            - Moving averages (short, medium, long-term)
            - Volatility measures
            - Support/Resistance
            - Volume indicators
            - Price returns
    """
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
    """
    Scale features to standardized ranges
    
    INPUT:
        features: pd.DataFrame of raw features
        method: str, 'standard' (mean=0, std=1), 'minmax' (0-1), or 'robust' (resistant to outliers)
    
    OUTPUT:
        Tuple of:
            - pd.DataFrame: Scaled features
            - Dict: Scaler object for later inverse transformation
    """
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
    """
    Analyze correlation between features and target variable
    
    INPUT:
        X: pd.DataFrame of features
        y: pd.Series of target variable (e.g., forward_return, binary_signal)
    
    OUTPUT:
        pd.DataFrame with columns:
            - feature: Feature name
            - correlation: Pearson correlation with target
            - p_value: Statistical significance
            - abs_correlation: Absolute correlation for ranking
    """
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
    """
    Remove highly correlated features to reduce multicollinearity
    
    INPUT:
        features: pd.DataFrame of features
        threshold: float, correlation threshold (default 0.95)
    
    OUTPUT:
        pd.DataFrame with redundant features removed
    """
    corr_matrix = features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Removing {len(to_drop)} collinear features: {to_drop}")
    return features.drop(columns=to_drop)


if __name__ == "__main__":
    print("Analytics Tools Module Loaded")
    print("Use: from tools import *")
