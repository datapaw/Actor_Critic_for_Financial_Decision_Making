# AI_agent_market_and_news

1. data download
2. first agent
3. data preparation

4. news agent
5. news checker
6. news sythetic generation

7. analyze titles finbert - optional comparison
8. main data collector

9. main agent
- INPUT:
    - **sp500_tools**date,model_prediction_period,model_confidence,time_horizon,primary_tools,secondary_tools,strategy,feature_rsi_14,feature_rsi_7,feature_macd,feature_macd_signal,feature_macd_histogram,feature_stoch_k,feature_stoch_d,feature_roc_12,feature_roc_5,feature_ema_5,feature_ema_12,feature_ema_20,feature_atr_14,feature_bb_position,feature_bb_width,feature_volume_sma,feature_volume_ratio,feature_trend_strength,close,volume,close_diff_next_day,feature_ema_26,feature_ema_50,feature_sma_20,feature_volatility_20,feature_resistance,feature_support,feature_midpoint,feature_cmf,feature_price_return_5d,feature_price_return_10d,feature_price_return_20d,feature_ema_12_26_diff,feature_ema_26_50_diff,feature_distance_to_resistance,feature_distance_to_support,feature_obv,feature_sma_50,feature_uptrend,feature_strong_uptrend,feature_resistance_60,feature_support_60,feature_volatility_60,feature_autocorr_60,feature_return_60d,feature_return_30d,feature_volume_avg_60,feature_sma_200,feature_sma_100,feature_price_sma20_diff,feature_price_sma50_diff,feature_resistance_30,feature_support_30,feature_volume_trend,feature_hurst_exponent
    - **scraped_news_sorted** 'date', 'date_only', 'source', 'title', 'sentiment', 'positive_score', 'negative_score', 'neutral_score', 'summary', 'link', 'is_synthetic', 'original_idx', 'Close', 'High', 'Low', 'Open', 'Volume', 'forward_return_10', 'forward_return_15', 'forward_return_30', 'forward_return_60', 'close_diff_next_day', 'target_period', 'forward_binary_10', 'forward_binary_15', 'forward_binary_30', 'forward_binary_60'

    - **sp500_tools_enriched** date,model_prediction_period,model_confidence,time_horizon,primary_tools,secondary_tools,strategy,feature_rsi_14,feature_rsi_7,feature_macd,feature_macd_signal,feature_macd_histogram,feature_stoch_k,feature_stoch_d,feature_roc_12,feature_roc_5,feature_ema_5,feature_ema_12,feature_ema_20,feature_atr_14,feature_bb_position,feature_bb_width,feature_volume_sma,feature_volume_ratio,feature_trend_strength,close,volume,close_diff_next_day,feature_ema_26,feature_ema_50,feature_sma_20,feature_volatility_20,feature_resistance,feature_support,feature_midpoint,feature_cmf,feature_price_return_5d,feature_price_return_10d,feature_price_return_20d,feature_ema_12_26_diff,feature_ema_26_50_diff,feature_distance_to_resistance,feature_distance_to_support,feature_obv,feature_sma_50,feature_uptrend,feature_strong_uptrend,feature_resistance_60,feature_support_60,feature_volatility_60,feature_autocorr_60,feature_return_60d,feature_return_30d,feature_volume_avg_60,feature_sma_200,feature_sma_100,feature_price_sma20_diff,feature_price_sma50_diff,feature_resistance_30,feature_support_30,feature_volume_trend,feature_hurst_exponent,date_only,news_count,avg_sentiment_val,avg_positive_score,avg_negative_score,avg_neutral_score,synthetic_frac,avg_title_len



TODO:
- merge files to what's necessary
- create main
