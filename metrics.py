"""
Financial Metrics Calculator
"""

from pathlib import Path
import pandas as pd
import numpy as np

RISK_FREE_RATE = 0.045  # 3.5% historical average (2001-2020)
# Note: Current rates (2024-2026) are higher (~5%), but using historical average
# for long-term backtest spanning 2001-2026

TEST_SET_SIZE = 941  # Last 941 days for test evaluation

CLASS_TO_HORIZON = {0: 10, 1: 15, 2: 30, 3: 60}  # Prediction class to days mapping


def load_sp500_data():
    """Load data/dataset/sp500_prep.csv with forward returns"""
    repo_root = Path(__file__).resolve().parent
    csv_path = repo_root / "data" / "dataset" / "sp500_prep.csv"
    
    df = pd.read_csv(csv_path)
    
    if "Date" in df.columns and "date" not in df.columns:
        df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    return df


def load_predictions():
    """Load results/chunked_main_votes.csv with ensemble_vote_pred"""
    repo_root = Path(__file__).resolve().parent
    csv_path = repo_root / "results" / "chunked_main_votes.csv"
    
    df = pd.read_csv(csv_path)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    return df


def compare_predictions_to_targets():
    """
    Compare ensemble predictions (ensemble_vote_pred) to actual best periods (target_period)
    
    What this does:
    1. Loads sp500_prep.csv which contains 'target_period' - the ACTUAL best horizon
       (0=10d, 1=15d, 2=30d, 3=60d) that had the highest forward return for each date
    
    2. Loads chunked_main_votes.csv which contains 'ensemble_vote_pred' - the MODEL's
       PREDICTION of which horizon will have the best return
    
    3. Merges them by date and compares:
       - Did the model predict the correct time horizon?
       - Accuracy = % of times prediction matched actual best period
    
    Returns:
        dict with:
        - accuracy: % correct predictions
        - total_samples: number of predictions
        - correct: number of correct predictions
        - merged_df: DataFrame with both columns for analysis
    """
    sp500_df = load_sp500_data()
    pred_df = load_predictions()
    
    # Merge by date to align predictions with actuals
    merged = pred_df.merge(sp500_df[["date", "target_period"]], on="date", how="inner")
    
    # Check if prediction matches actual best period
    merged["correct"] = (merged["ensemble_vote_pred"] == merged["target_period"]).astype(int)
    
    # Calculate accuracy
    total_samples = len(merged)
    correct_predictions = merged["correct"].sum()
    accuracy = (correct_predictions / total_samples * 100) if total_samples > 0 else 0.0
    
    print("="*80)
    print("PREDICTION VS ACTUAL TARGET PERIOD COMPARISON")
    print("="*80)
    print(f"\nTotal samples:         {total_samples}")
    print(f"Correct predictions:   {correct_predictions}")
    print(f"Accuracy:              {accuracy:.2f}%")
    print(f"\nThis measures: Did the model correctly predict which time horizon")
    print(f"               (10/15/30/60 days) would have the best forward return?")
    print("="*80 + "\n")
    
    return {
        "accuracy": accuracy,
        "total_samples": total_samples,
        "correct": correct_predictions,
        "incorrect": total_samples - correct_predictions,
        "merged_df": merged
    }


def get_test_set():
    """
    Get last 941 rows as test set
    
    Returns:
        tuple: (sp500_test, predictions_test) - both DataFrames filtered to test period
    """
    sp500_df = load_sp500_data()
    pred_df = load_predictions()
    
    # Take last 941 rows from sp500 data
    sp500_test = sp500_df.tail(TEST_SET_SIZE).reset_index(drop=True)
    
    # Match predictions to test period dates
    test_dates = sp500_test["date"]
    pred_test = pred_df[pred_df["date"].isin(test_dates)].reset_index(drop=True)
    
    print(f"✓ Test set: {len(sp500_test)} rows")
    print(f"  Date range: {sp500_test['date'].min()} to {sp500_test['date'].max()}")
    print(f"  Predictions: {len(pred_test)} rows")
    
    return sp500_test, pred_test


def extract_actual_returns(merged_df):
    """
    Extract actual forward return as PERCENTAGE based on what the model predicted
    
    For each row:
    - If ensemble_vote_pred = 0 (predicted 10 days best) → get forward_return_10
    - If ensemble_vote_pred = 1 (predicted 15 days best) → get forward_return_15
    - If ensemble_vote_pred = 2 (predicted 30 days best) → get forward_return_30
    - If ensemble_vote_pred = 3 (predicted 60 days best) → get forward_return_60
    
    IMPORTANT: forward_return columns store AVERAGE POINTS PER DAY
    From data_preparation.py: forward_return_N = (Close_future - Close_now) / N
    
    To get actual percentage return:
    1. Multiply by N to get total point change
    2. Divide by Close to get percentage
    
    Example: Close=1500, forward_return_10=4.4 (avg +4.4 pts/day)
    - Total points = 4.4 * 10 = 44
    - Percentage = 44 / 1500 * 100 = 2.93%
    
    Args:
        merged_df: DataFrame with ensemble_vote_pred, forward_return_*, and Close columns
    
    Returns:
        np.array of percentage returns
    """
    returns_pct = []
    
    for _, row in merged_df.iterrows():
        pred_class = int(row["ensemble_vote_pred"])
        horizon = CLASS_TO_HORIZON[pred_class]
        
        # forward_return columns store AVERAGE POINTS PER DAY
        # From data_preparation.py: (Close_future - Close_now) / days
        # Example: forward_return_10 = 4.4 means +4.4 points/day average
        # To get percentage: (forward_return * days) / Close * 100
        forward_return_per_day = row[f"forward_return_{horizon}"]
        close_price = row["Close"]
        
        if pd.notna(forward_return_per_day) and pd.notna(close_price) and close_price != 0:
            total_points = forward_return_per_day * horizon  # Convert to total point change
            return_pct = (total_points / close_price) * 100  # Convert to percentage
        else:
            return_pct = np.nan
        
        returns_pct.append(return_pct)
    
    return np.array(returns_pct)


def calculate_sharpe_ratio(returns, predictions):
    """
    Calculate annualized Sharpe Ratio with PERCENTAGE returns
    
    Formula: (Mean Return - Risk Free Rate) * sqrt(Periods/Year) / Std Dev of Returns
    
    Now properly uses percentage returns (e.g., 2.5% not 25 points)
    
    Steps:
    1. Calculate average holding period from predictions
    2. Calculate periods per year (252 trading days / avg holding days)
    3. Convert annual risk-free rate to per-period rate
    4. Calculate excess return and annualize
    
    Args:
        returns: numpy array of PERCENTAGE returns from strategy
        predictions: numpy array of prediction classes (0/1/2/3)
    
    Returns:
        dict with sharpe_ratio and related metrics
    """
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return {"sharpe_ratio": 0.0, "avg_holding_days": 0, "periods_per_year": 0}
    
    # Calculate average holding period from predictions
    holding_days = [CLASS_TO_HORIZON[int(pred)] for pred in predictions]
    avg_holding_days = np.mean(holding_days)
    
    # Calculate how many periods per year
    periods_per_year = 252 / avg_holding_days
    
    # Convert annual risk-free rate (4.5%) to per-period percentage rate
    period_risk_free_pct = (RISK_FREE_RATE * 100) / periods_per_year
    
    # Calculate excess return (return above risk-free)
    excess_return = mean_return - period_risk_free_pct
    
    # Annualized Sharpe Ratio
    sharpe_annualized = (excess_return * np.sqrt(periods_per_year)) / std_return
    
    return {
        "sharpe_ratio": sharpe_annualized,
        "avg_holding_days": avg_holding_days,
        "periods_per_year": periods_per_year,
        "period_risk_free_pct": period_risk_free_pct,
        "excess_return": excess_return,
        "mean_return": mean_return,
        "std_return": std_return
    }


def calculate_maximum_drawdown(returns):
    """
    Calculate Maximum Drawdown - largest peak-to-trough decline
    
    Shows the worst-case loss from any peak in portfolio value.
    
    Args:
        returns: numpy array of percentage returns
    
    Returns:
        dict with max_drawdown (as percentage, negative value)
    """
    # Build cumulative portfolio value (starting at 1.0)
    cumulative = np.cumprod(1 + returns / 100)
    
    # Track running maximum (peak values)
    running_max = np.maximum.accumulate(cumulative)
    
    # Calculate drawdown at each point
    drawdown = (cumulative - running_max) / running_max * 100
    
    # Maximum drawdown (most negative value)
    max_dd = np.min(drawdown)
    
    # Find peak and trough indices
    trough_idx = np.argmin(drawdown)
    peak_idx = np.argmax(cumulative[:trough_idx]) if trough_idx > 0 else 0
    
    return {
        "max_drawdown_pct": max_dd,
        "peak_idx": int(peak_idx),
        "trough_idx": int(trough_idx)
    }


def calculate_sortino_ratio(returns, predictions):
    """
    Calculate Sortino Ratio - like Sharpe but only penalizes downside volatility
    
    Better than Sharpe for strategies with asymmetric returns because it doesn't
    penalize upside volatility.
    
    Args:
        returns: numpy array of percentage returns
        predictions: numpy array of prediction classes
    
    Returns:
        dict with sortino_ratio
    """
    mean_return = np.mean(returns)
    
    # Only look at negative returns for downside deviation
    negative_returns = returns[returns < 0]
    
    if len(negative_returns) == 0:
        return {"sortino_ratio": np.inf, "downside_std": 0.0}
    
    downside_std = np.std(negative_returns)
    
    if downside_std == 0:
        return {"sortino_ratio": 0.0, "downside_std": 0.0}
    
    # Calculate average holding period
    holding_days = [CLASS_TO_HORIZON[int(pred)] for pred in predictions]
    avg_holding_days = np.mean(holding_days)
    periods_per_year = 252 / avg_holding_days
    
    # Convert annual risk-free to period percentage
    period_risk_free_pct = (RISK_FREE_RATE * 100) / periods_per_year
    
    # Excess return
    excess_return = mean_return - period_risk_free_pct
    
    # Annualized Sortino
    sortino = (excess_return * np.sqrt(periods_per_year)) / downside_std
    
    return {
        "sortino_ratio": sortino,
        "downside_std": downside_std
    }


def calculate_annualized_return(returns, predictions):
    """
    Calculate Annualized Return - geometric mean return per year
    
    Normalizes returns to yearly basis for comparison.
    
    Args:
        returns: numpy array of percentage returns
        predictions: numpy array of prediction classes
    
    Returns:
        dict with annualized_return_pct
    """
    # Calculate total compounded return
    cumulative_return = np.prod(1 + returns / 100) - 1
    
    # Calculate time period in years
    holding_days = [CLASS_TO_HORIZON[int(pred)] for pred in predictions]
    avg_holding_days = np.mean(holding_days)
    periods_per_year = 252 / avg_holding_days
    years = len(returns) / periods_per_year
    
    if years == 0 or (1 + cumulative_return) <= 0:
        return {"annualized_return_pct": 0.0, "total_return_pct": 0.0}
    
    # Annualize
    annualized = ((1 + cumulative_return) ** (1 / years) - 1) * 100
    
    return {
        "annualized_return_pct": annualized,
        "total_return_pct": cumulative_return * 100,
        "years": years
    }


def calculate_calmar_ratio(annualized_return_pct, max_drawdown_pct):
    """
    Calculate Calmar Ratio - return per unit of worst-case risk
    
    Formula: Annualized Return / |Maximum Drawdown|
    
    Args:
        annualized_return_pct: annual return percentage
        max_drawdown_pct: maximum drawdown percentage (negative)
    
    Returns:
        dict with calmar_ratio
    """
    if max_drawdown_pct >= 0 or abs(max_drawdown_pct) < 0.001:
        return {"calmar_ratio": 0.0}
    
    calmar = annualized_return_pct / abs(max_drawdown_pct)
    
    return {"calmar_ratio": calmar}


def calculate_win_rate(returns):
    """
    Calculate Win Rate and related statistics
    
    Measures how often predictions are profitable and average win/loss sizes.
    
    Args:
        returns: numpy array of percentage returns
    
    Returns:
        dict with win_rate, profit_factor, etc.
    """
    total_trades = len(returns)
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns <= 0]
    
    wins = len(winning_trades)
    losses = len(losing_trades)
    
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    
    # Average win and loss
    avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
    avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0
    
    # Profit factor (net definition: shows net profit per dollar lost)
    total_gains = np.sum(winning_trades) if len(winning_trades) > 0 else 0.0
    total_losses = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0.0
    # Net profit factor = (gains - losses) / losses
    # This shows how much net profit you make per dollar lost
    profit_factor = (total_gains - total_losses) / total_losses if total_losses > 0 else 0.0
    
    return {
        "win_rate": win_rate,
        "total_trades": total_trades,
        "winning_trades": wins,
        "losing_trades": losses,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TEST SET EVALUATION - ALL FINANCIAL METRICS")
    print("="*80 + "\n")
    
    # Get test set
    sp500_test, pred_test = get_test_set()
    
    # Merge them
    merged = pred_test.merge(sp500_test, on="date", how="inner")
    print(f"✓ Merged test data: {len(merged)} rows\n")
    
    # Extract actual returns based on predictions
    actual_returns = extract_actual_returns(merged)
    predictions = merged["ensemble_vote_pred"].values
    
    # Check for NaN values (happens when forward returns not available for recent dates)
    valid_mask = ~np.isnan(actual_returns)
    n_nan = np.sum(~valid_mask)
    
    if n_nan > 0:
        print(f"⚠️  Warning: {n_nan} rows have NaN returns (forward data not available)")
        print(f"   This is normal for recent dates - filtering them out...")
        actual_returns = actual_returns[valid_mask]
        predictions = predictions[valid_mask]
        print(f"   Using {len(actual_returns)} valid rows for calculation\n")
    
    if len(actual_returns) == 0:
        print("❌ Error: No valid returns found in test set!")
        print("   The test set is too recent - forward returns not available yet.")
        print("   Try using an earlier test period or wait for more data.")
        import sys
        sys.exit(1)
    
    print(f"✓ Extracted {len(actual_returns)} actual returns based on predictions")
    print(f"  Mean return: {np.mean(actual_returns):.4f}%")
    print(f"  Std dev:     {np.std(actual_returns):.4f}%")
    print(f"  Min return:  {np.min(actual_returns):.4f}%")
    print(f"  Max return:  {np.max(actual_returns):.4f}%\n")
    
    # Calculate all metrics
    sharpe_metrics = calculate_sharpe_ratio(actual_returns, predictions)
    mdd_metrics = calculate_maximum_drawdown(actual_returns)
    sortino_metrics = calculate_sortino_ratio(actual_returns, predictions)
    annual_metrics = calculate_annualized_return(actual_returns, predictions)
    calmar_metrics = calculate_calmar_ratio(annual_metrics['annualized_return_pct'], 
                                            mdd_metrics['max_drawdown_pct'])
    win_metrics = calculate_win_rate(actual_returns)
    
    # Display results
    print("="*80)
    print("FINANCIAL METRICS SUMMARY")
    print("="*80)
    
    print("\n📊 RISK-ADJUSTED RETURNS")
    print("-" * 80)
    print(f"Sharpe Ratio:              {sharpe_metrics['sharpe_ratio']:>10.4f}")
    print(f"Sortino Ratio:             {sortino_metrics['sortino_ratio']:>10.4f}")
    print(f"Calmar Ratio:              {calmar_metrics['calmar_ratio']:>10.4f}")
    
    print("\n📉 RISK METRICS")
    print("-" * 80)
    print(f"Maximum Drawdown:          {mdd_metrics['max_drawdown_pct']:>10.2f}%")
    
    print("\n💰 RETURN METRICS")
    print("-" * 80)
    print(f"Annualized Return:         {annual_metrics['annualized_return_pct']:>10.2f}%")
    print(f"Total Return:              {annual_metrics['total_return_pct']:>10.2f}%")
    print(f"Time Period:               {annual_metrics['years']:>10.2f} years")
    print(f"Risk-Free Rate (annual):   {RISK_FREE_RATE*100:>10.2f}%")
    
    print("\n🎯 WIN STATISTICS")
    print("-" * 80)
    print(f"Win Rate:                  {win_metrics['win_rate']:>10.2f}%")
    print(f"Total Trades:              {win_metrics['total_trades']:>10d}")
    print(f"  Winning Trades:          {win_metrics['winning_trades']:>10d}")
    print(f"  Losing Trades:           {win_metrics['losing_trades']:>10d}")
    print(f"Average Win:               {win_metrics['avg_win']:>10.4f}%")
    print(f"Average Loss:              {win_metrics['avg_loss']:>10.4f}%")
    print(f"Profit Factor:             {win_metrics['profit_factor']:>10.4f}")
    
    print("\n📈 HOLDING PERIOD")
    print("-" * 80)
    print(f"Average holding days:      {sharpe_metrics['avg_holding_days']:>10.2f}")
    print(f"Periods per year:          {sharpe_metrics['periods_per_year']:>10.2f}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    sharpe = sharpe_metrics['sharpe_ratio']
    print(f"\nSharpe Ratio ({sharpe:.4f}):")
    if sharpe < 0:
        print(f"  ❌ Negative - Losing money or not beating risk-free rate")
    elif sharpe < 0.5:
        print(f"  ⚠️  Very Poor - Barely compensating for risk")
    elif sharpe < 0.9:
        print(f"  ⚠️  Below Benchmark - Underperforming buy-and-hold S&P 500")
    elif sharpe < 1.0:
        print(f"  ✓  Decent - Matching or slightly beating S&P 500 benchmark (0.5-0.9)")
    elif sharpe < 1.5:
        print(f"  ✓✓ Good - Solid risk-adjusted returns above benchmark")
    elif sharpe < 2.0:
        print(f"  ✓✓ Very Good - Professional level performance")
    else:
        print(f"  ✓✓✓ Excellent - Institutional quality (verify not overfitting)")
    
    sortino = sortino_metrics['sortino_ratio']
    print(f"\nSortino Ratio ({sortino:.4f}):")
    print(f"  Better than Sharpe if asymmetric returns (limits losses > gains)")
    if sortino > sharpe * 1.2:
        print(f"  ✓ Strategy protects downside well (Sortino >> Sharpe)")
    
    win_rate = win_metrics['win_rate']
    print(f"\nWin Rate ({win_rate:.2f}%):")
    if win_rate < 50:
        print(f"  ⚠️  Below 50% - More losses than wins")
    elif win_rate < 55:
        print(f"  ✓  Baseline - Slightly better than random")
    elif win_rate < 60:
        print(f"  ✓✓ Good - Solid prediction accuracy")
    else:
        print(f"  ✓✓✓ Excellent - Very high accuracy (check for data leakage)")
    
    pf = win_metrics['profit_factor']
    print(f"\nProfit Factor ({pf:.4f}):")
    if pf < 1.0:
        print(f"  ❌ Losing money overall (gains < losses)")
    elif pf < 1.5:
        print(f"  ✓  Slightly profitable")
    elif pf < 2.0:
        print(f"  ✓✓ Good profitability")
    else:
        print(f"  ✓✓✓ Excellent profitability")
    
    print("\n" + "="*80)
