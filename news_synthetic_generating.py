# Synthetic news generator
# We only have real news for Dec 2025 - Jan 2026, so we need to generate
# synthetic news for the rest of the historical period (2001-2025)
# Uses statistical patterns from real news + correlations with price volatility

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# file paths
REAL_NEWS_PATH = Path('delete/scraped_news_sorted.csv')
PRICES_PATH = Path('data/dataset/sp500_prep.csv')
OUTPUT_PATH = Path('data/dataset/news_sp500_synthetic_full.csv')


def analyze_real_news(news_path):
    # pull stats from the real news we have
    df = pd.read_csv(news_path)
    
    print("="*70)
    print("ANALYZING REAL NEWS DATA")
    print("="*70)
    print(f"Total articles: {len(df)}")
    
    # get sentiment statistics
    stats = {
        'mean_positive': df['positive_score'].mean(),
        'std_positive': df['positive_score'].std(),
        'mean_negative': df['negative_score'].mean(),
        'std_negative': df['negative_score'].std(),
        'mean_neutral': df['neutral_score'].mean(),
        'std_neutral': df['neutral_score'].std(),
        'mean_overall': (df['positive_score'] - df['negative_score']).mean(),
        'std_overall': (df['positive_score'] - df['negative_score']).std(),
    }
    
    # check how many articles per day on average
    df['date_only'] = pd.to_datetime(df['date']).dt.normalize()
    daily_counts = df.groupby('date_only').size()
    
    stats['mean_news_count'] = daily_counts.mean()
    stats['std_news_count'] = daily_counts.std()
    
    print(f"\nSentiment Statistics:")
    print(f"  Positive score: {stats['mean_positive']:.4f} ± {stats['std_positive']:.4f}")
    print(f"  Negative score: {stats['mean_negative']:.4f} ± {stats['std_negative']:.4f}")
    print(f"  Overall sentiment: {stats['mean_overall']:.4f} ± {stats['std_overall']:.4f}")
    print(f"  News per day: {stats['mean_news_count']:.2f} ± {stats['std_news_count']:.2f}")
    
    return stats, daily_counts


def load_historical_volatility(prices_path):
    """Load prices and calculate daily volatility."""
    df = pd.read_csv(prices_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Calculate returns and volatility
    df['return'] = df['Close'].pct_change()
    df['volatility_5d'] = df['return'].rolling(5).std()
    
    return df[['Close', 'return', 'volatility_5d']].copy()


def synthesize_news_for_historical_dates(stats, prices_df, real_dates):
    """Generate article-level synthetic news for historical dates using only statistics.

    This function does not sample or depend on any real article text. It creates
    per-article synthetic rows by sampling sentiment scores and counts from the
    empirical distributions (with volatility/return-based biases).
    """
    print("\n" + "="*70)
    print("GENERATING ARTICLE-LEVEL SYNTHETIC NEWS (statistical)")
    print("="*70)

    all_dates = prices_df.index
    real_date_set = set(pd.to_datetime(list(real_dates)).normalize())

    synthetic_articles = []

    # Build base distributions
    mean_pos = stats.get('mean_positive', 0.2)
    std_pos = max(1e-6, stats.get('std_positive', 0.05))
    mean_neg = stats.get('mean_negative', 0.1)
    std_neg = max(1e-6, stats.get('std_negative', 0.05))
    mean_neu = stats.get('mean_neutral', 0.7)
    std_neu = max(1e-6, stats.get('std_neutral', 0.05))

    for date in all_dates:
        date_norm = pd.to_datetime(date).normalize()
        if date_norm in real_date_set:
            continue

        vol = prices_df.loc[date, 'volatility_5d']
        ret = prices_df.loc[date, 'return']
        if pd.isna(vol):
            vol = stats.get('std_overall', 0.01)

        vol_norm = float(max(0, min(1, vol / (2 * max(1e-6, stats.get('std_overall', 0.01))))))

        # Expected articles count increases with volatility
        lam = max(0.5, stats.get('mean_news_count', 1.0) + vol_norm * (stats.get('std_news_count', 1.0)))
        news_count = int(np.random.poisson(lam=lam))
        news_count = max(1, news_count)

        for n in range(news_count):
            # sample sentiment components from normals, add bias from return
            pos = np.random.normal(loc=mean_pos + 0.5 * max(0, ret), scale=std_pos * (1 + vol_norm))
            neg = np.random.normal(loc=mean_neg + 0.5 * max(0, -ret), scale=std_neg * (1 + vol_norm))
            neu = np.random.normal(loc=mean_neu, scale=std_neu * (1 - 0.5 * vol_norm))

            pos = float(np.clip(pos, 0, 1))
            neg = float(np.clip(neg, 0, 1))
            neu = float(np.clip(neu, 0, 1))

            s = pos + neg + neu
            if s <= 0:
                pos, neg, neu = mean_pos, mean_neg, mean_neu
                s = pos + neg + neu
            pos, neg, neu = pos / s, neg / s, neu / s

            hour = np.random.randint(9, 16)
            minute = np.random.randint(0, 60)
            synth_datetime = pd.to_datetime(date_norm) + pd.Timedelta(hours=hour, minutes=minute)

            # create template-like title and summary to mimic real format
            sentiment_label = 'positive' if pos - neg > 0.05 else ('negative' if neg - pos > 0.05 else 'neutral')
            title = f"Market {sentiment_label.capitalize()}: synthetic headline"
            summary = f"Synthetic summary ({sentiment_label}) for {date_norm.date()}"

            article = {
                'date': synth_datetime,
                'source': 'synthetic_pool',
                'title': title,
                'sentiment': sentiment_label,
                'positive_score': pos,
                'negative_score': neg,
                'neutral_score': neu,
                'summary': summary,
                'link': f'synthetic://{date_norm.date()}_{n}',
                'is_synthetic': True,
                'original_idx': -1
            }
            synthetic_articles.append(article)

    print(f"Generated {len(synthetic_articles)} synthetic article-level rows")
    return pd.DataFrame(synthetic_articles)


def main():
    # Load and analyze real news
    real_news = pd.read_csv(REAL_NEWS_PATH)
    stats, real_daily_counts = analyze_real_news(REAL_NEWS_PATH)

    # Load historical prices and volatility
    prices_df = load_historical_volatility(PRICES_PATH)

    # Generate synthetic article-level news based on statistical distributions
    synthetic_articles = synthesize_news_for_historical_dates(stats, prices_df, real_news['date'])

    # Prepare real articles in matching format
    real_articles = real_news.copy()
    real_articles['date'] = pd.to_datetime(real_articles['date'])
    # Ensure columns exist
    for c in ['source', 'title', 'sentiment', 'positive_score', 'negative_score', 'neutral_score', 'summary', 'link']:
        if c not in real_articles.columns:
            real_articles[c] = None
    real_articles['is_synthetic'] = False
    real_articles['original_idx'] = -1

    # Select and reorder columns to match synthetic
    cols = ['date', 'source', 'title', 'sentiment', 'positive_score', 'negative_score', 'neutral_score', 'summary', 'link', 'is_synthetic', 'original_idx']
    real_articles = real_articles[cols]

    # Combine article-level datasets
    combined_articles = pd.concat([real_articles, synthetic_articles], ignore_index=True)
    combined_articles = combined_articles.sort_values('date').reset_index(drop=True)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_articles.to_csv(OUTPUT_PATH, index=False)

    print(f"\nTotal article rows: {len(combined_articles)}")
    print(f"Real article rows: {(combined_articles['is_synthetic'] == False).sum()}")
    print(f"Synthetic article rows: {(combined_articles['is_synthetic'] == True).sum()}")
    print(f"\nOutput saved to: {OUTPUT_PATH}")

    # Summary statistics
    print("\n" + "="*70)
    print("FINAL DATASET SUMMARY")
    print("="*70)
    print(f"Date range: {combined_articles['date'].min()} to {combined_articles['date'].max()}")
    print(f"\nSample real article counts per day (mean): {real_daily_counts.mean():.2f}")


if __name__ == '__main__':
    main()
