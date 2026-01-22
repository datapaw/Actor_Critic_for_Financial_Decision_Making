"""
Compute summed forward returns where binary flags equal 1 and evaluate returns by
ensemble vote predictions.

Sources:
- data/dataset/sp500_prep.csv with forward_return_* and forward_binary_* columns
- results/chunked_main_votes.csv with ensemble_vote_pred and date

Outputs:
- results/forward_binary_return_sums.csv (binary-flag summary)
- results/chunked_vote_return_summary.csv (per-class vote return stats, if votes file exists)
- results/chunked_vote_returns.csv (merged rows with predicted_return, if votes file exists)
"""
from pathlib import Path
import pandas as pd


VOTE_PATH = Path(__file__).resolve().parent / "results" / "chunked_main_votes.csv"
CRITIC_TIMING_PATH = Path(__file__).resolve().parent / "results" / "critic_timing_results_lstm_v2.csv"
CLASS_TO_HORIZON = {0: 10, 1: 15, 2: 30, 3: 60}
LABEL_MAPPING = {"immediate": 0, "wait_2_days": 1}


def load_prep_csv(csv_path: str | Path = None) -> pd.DataFrame:
    repo_root = Path(__file__).resolve().parent
    default_path = repo_root / "data" / "dataset" / "sp500_prep.csv"
    csv_path = Path(csv_path) if csv_path else default_path

    if not csv_path.exists():
        raise FileNotFoundError(f"sp500_prep.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)
    # normalize date column
    if "Date" in df.columns and "date" not in df.columns:
        df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_votes(csv_path: str | Path = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else VOTE_PATH
    if not path.exists():
        raise FileNotFoundError(f"chunked_main_votes.csv not found at {path}")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_critic_timing(csv_path: str | Path = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else CRITIC_TIMING_PATH
    if not path.exists():
        raise FileNotFoundError(f"critic_timing_results_lstm_v2.csv not found at {path}")
    df = pd.read_csv(path)
    # Take only last 941 rows
    df = df.tail(941).reset_index(drop=True)
    # Add date column from sp500_prep.csv
    prep_df = load_prep_csv()
    date_col = prep_df["date"].tail(941).reset_index(drop=True)
    df.insert(0, "date", date_col)
    # Map critic_timing_label if it exists
    if "critic_timing_label" in df.columns:
        df["critic_timing_label"] = df["critic_timing_label"].map(LABEL_MAPPING)
    return df


def compute_sums(df: pd.DataFrame) -> pd.DataFrame:
    horizons = [10, 15, 30, 60]
    rows = []

    for h in horizons:
        ret_col = f"forward_return_{h}"
        bin_col = f"forward_binary_{h}"

        if ret_col not in df.columns or bin_col not in df.columns:
            rows.append({
                "horizon": h,
                "count_flag_1": 0,
                "total_return_flag_1": 0.0,
                "mean_return_flag_1": 0.0,
            })
            continue

        mask = df[bin_col] == 1
        flagged = df.loc[mask, ret_col].astype(float)

        count_ones = int(mask.sum())
        total_return = float(flagged.sum()) if count_ones > 0 else 0.0
        mean_return = float(flagged.mean()) if count_ones > 0 else 0.0

        positives = flagged[flagged > 0]
        negatives = flagged[flagged < 0]
        pos_sum = float(positives.sum()) if not positives.empty else 0.0
        neg_sum = float(negatives.sum()) if not negatives.empty else 0.0
        pos_count = int(len(positives))
        neg_count = int(len(negatives))

        rows.append({
            "horizon": h,
            "count_flag_1": count_ones,
            "total_return_flag_1": total_return,
            "mean_return_flag_1": mean_return,
            "positive_count": pos_count,
            "positive_sum": pos_sum,
            "negative_count": neg_count,
            "negative_sum": neg_sum,
        })

    return pd.DataFrame(rows)


def compute_vote_returns(votes_df: pd.DataFrame, prep_df: pd.DataFrame) -> pd.DataFrame:
    merged = votes_df.merge(prep_df, on="date", how="inner", suffixes=("_vote", "_prep"))
    merged["predicted_horizon_days"] = merged["ensemble_vote_pred"].map(CLASS_TO_HORIZON).astype("Int64")

    def pick_return(row):
        horizon = row["predicted_horizon_days"]
        if pd.isna(horizon):
            return pd.NA
        col = f"forward_return_{int(horizon)}"
        return row.get(col, pd.NA)

    merged["predicted_return"] = merged.apply(pick_return, axis=1)
    return merged


def summarize_vote_returns(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls, horizon in CLASS_TO_HORIZON.items():
        subset = merged[merged["ensemble_vote_pred"] == cls]
        preds = subset["predicted_return"].astype(float)
        if len(preds) == 0:
            rows.append({
                "class": cls,
                "horizon_days": horizon,
                "count": 0,
                "sum_return": 0.0,
                "mean_return": 0.0,
                "win_rate_pct": 0.0,
                "positive_count": 0,
                "positive_sum": 0.0,
                "negative_count": 0,
                "negative_sum": 0.0,
            })
            continue

        positives = preds[preds > 0]
        negatives = preds[preds < 0]

        rows.append({
            "class": cls,
            "horizon_days": horizon,
            "count": len(preds),
            "sum_return": float(preds.sum()),
            "mean_return": float(preds.mean()),
            "win_rate_pct": float((preds > 0).mean() * 100),
            "positive_count": int(len(positives)),
            "positive_sum": float(positives.sum()) if len(positives) else 0.0,
            "negative_count": int(len(negatives)),
            "negative_sum": float(negatives.sum()) if len(negatives) else 0.0,
        })
    return pd.DataFrame(rows)


def summarize_critic_timing_returns(merged: pd.DataFrame) -> pd.DataFrame:
    if "critic_timing_label" not in merged.columns:
        return pd.DataFrame()

    rows = []
    labels = merged["critic_timing_label"].dropna().unique()
    
    # Create a lookup for future forward returns (date + 2 days)
    future_lookup = merged.copy()
    future_lookup["date"] = future_lookup["date"] + pd.Timedelta(days=2)
    
    for label in sorted(labels):
        if label not in [0, 1]:
            print(f"{int(label)}")
            continue
        
        subset = merged[merged["critic_timing_label"] == label].copy()
        
        if label == 1:
            # For label=1, use forward returns from 2 days in the future
            future_cols = ["date", "predicted_horizon_days"] + [f"forward_return_{h}" for h in [10, 15, 30, 60]]
            subset = subset.merge(
                future_lookup[future_cols],
                on=["date", "predicted_horizon_days"],
                suffixes=("", "_future"),
                how="left"
            )
            # Replace the forward returns with future values
            for h in [10, 15, 30, 60]:
                subset[f"forward_return_{h}"] = subset[f"forward_return_{h}_future"]
                subset.drop(f"forward_return_{h}_future", axis=1, inplace=True)
            
            # Recalculate predicted_return
            def pick_return(row):
                horizon = row["predicted_horizon_days"]
                if pd.isna(horizon):
                    return pd.NA
                col = f"forward_return_{int(horizon)}"
                return row.get(col, pd.NA)
            
            subset["predicted_return"] = subset.apply(pick_return, axis=1)
        
        preds = subset["predicted_return"].astype(float)
        if len(preds) == 0:
            rows.append({
                "critic_timing_label": int(label),
                "count": 0,
                "sum_return": 0.0,
                "mean_return": 0.0,
                "win_rate_pct": 0.0,
                "positive_count": 0,
                "positive_sum": 0.0,
                "negative_count": 0,
                "negative_sum": 0.0,
            })
            continue

        positives = preds[preds > 0]
        negatives = preds[preds < 0]

        rows.append({
            "critic_timing_label": int(label),
            "count": len(preds),
            "sum_return": float(preds.sum()),
            "mean_return": float(preds.mean()),
            "win_rate_pct": float((preds > 0).mean() * 100),
            "positive_count": int(len(positives)),
            "positive_sum": float(positives.sum()) if len(positives) else 0.0,
            "negative_count": int(len(negatives)),
            "negative_sum": float(negatives.sum()) if len(negatives) else 0.0,
        })
    
    return pd.DataFrame(rows)


def summarize_predicted_returns_by_horizon(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in sorted(CLASS_TO_HORIZON.values()):
        subset = merged[merged["predicted_horizon_days"] == horizon]
        preds = subset["predicted_return"].astype(float)
        if len(preds) == 0:
            rows.append({
                "horizon": horizon,
                "count_pred": 0,
                "total_return_pred": 0.0,
                "mean_return_pred": 0.0,
                "positive_count_pred": 0,
                "positive_sum_pred": 0.0,
                "negative_count_pred": 0,
                "negative_sum_pred": 0.0,
            })
            continue

        positives = preds[preds > 0]
        negatives = preds[preds < 0]

        rows.append({
            "horizon": horizon,
            "count_pred": len(preds),
            "total_return_pred": float(preds.sum()),
            "mean_return_pred": float(preds.mean()),
            "positive_count_pred": int(len(positives)),
            "positive_sum_pred": float(positives.sum()) if len(positives) else 0.0,
            "negative_count_pred": int(len(negatives)),
            "negative_sum_pred": float(negatives.sum()) if len(negatives) else 0.0,
        })
    return pd.DataFrame(rows)


def main(csv_path: str | Path | None = None):
    df = load_prep_csv(csv_path)
    n_eval = min(len(df), 941)
    df_eval = df.tail(n_eval)
    summary = compute_sums(df_eval)

    print(f"Using last {n_eval} rows (tail of prep data) for evaluation")
    print("Forward return sums where binary flag == 1")
    print(summary.to_string(index=False))

    # Optional: evaluate returns by ensemble vote predictions
    try:
        votes_df = load_votes()
        critic_timing_df = None
        try:
            critic_timing_df = load_critic_timing()
        except FileNotFoundError:
            print("Critic timing data not found; skipping critic timing merge")
        
        merged = compute_vote_returns(votes_df, df_eval)
        
        # Merge critic timing data if available
        if critic_timing_df is not None:
            critic_subset = critic_timing_df[["date", "critic_timing_label"]].copy()
            merged = merged.merge(critic_subset, on="date", how="left")
        
        vote_summary = summarize_vote_returns(merged)

        print("\nPredicted returns by ensemble_vote_pred (mapped to 60/30/15/10 days):")
        print(vote_summary.to_string(index=False))
        
        # Print critic timing summary if available
        if critic_timing_df is not None:
            critic_summary = summarize_critic_timing_returns(merged)
            if not critic_summary.empty:
                print("\nPredicted returns by critic_timing_label (0=immediate, 1=wait_2_days):")
                print(critic_summary.to_string(index=False))
    except FileNotFoundError as e:
        print(f"\nVote return analysis skipped: {e}")


if __name__ == "__main__":
    main()
