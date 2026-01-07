"""Analyze titles with FinBERT and compare to existing abstract sentiment.

Produces:
- data/dataset/title_finbert_daily_compare.csv
- data/dataset/title_vs_abstract_sentiment.png

Usage:
  pip install -r requirements.txt  # see run instructions
  python analyze_titles_finbert.py --input delete/scraped_news_sorted.csv

The script attempts to detect the existing abstract sentiment column and will run
FinBERT only on the `title` column (or a custom title column).
"""
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from transformers import pipeline
except Exception:
    pipeline = None


DEFAULT_INPUT = Path('delete/scraped_news_sorted.csv')
OUT_CSV = Path('data/dataset/title_finbert_daily_compare.csv')
OUT_PNG = Path('data/dataset/title_vs_abstract_sentiment.png')


def detect_sentiment_column(df: pd.DataFrame):
    # Common name guesses
    candidates = ['sentiment', 'sentiment_abstract', 'abstract_sentiment', 'sentiment_text']
    for c in candidates:
        if c in df.columns:
            return c
    # last resort: columns with small set of string values
    for c in df.columns:
        if df[c].dtype == object:
            vals = df[c].dropna().unique()[:10]
            if all(isinstance(v, str) for v in vals) and set([v.lower() for v in vals]).intersection({'positive','negative','neutral'}):
                return c
    return None


def map_sentiment_label(label: str) -> str:
    if not isinstance(label, str):
        return 'neutral'
    l = label.lower()
    if 'pos' in l:
        return 'positive'
    if 'neg' in l:
        return 'negative'
    if 'neu' in l:
        return 'neutral'
    # some FinBERT variants return labels like LABEL_0; try mapping via scores
    return label


def run_finbert_on_titles(titles: pd.Series, model_name: str = 'yiyanghkust/finbert-tone', batch_size: int = 32):
    if pipeline is None:
        raise RuntimeError('transformers not available — install with `pip install transformers torch`')

    classifier = pipeline('sentiment-analysis', model=model_name)

    labels = []
    for i in range(0, len(titles), batch_size):
        batch = titles.iloc[i:i+batch_size].astype(str).tolist()
        preds = classifier(batch)
        # preds: list of dicts with 'label' and 'score'
        for p in preds:
            labels.append(map_sentiment_label(p.get('label', 'neutral')))
    return labels


def aggregate_daily(df: pd.DataFrame, sentiment_col: str, date_col: str = 'date') -> pd.DataFrame:
    df = df.copy()
    # create normalized date
    if date_col in df.columns:
        df['date_only'] = pd.to_datetime(df[date_col]).dt.normalize()
    elif 'date_only' in df.columns:
        df['date_only'] = pd.to_datetime(df['date_only']).dt.normalize()
    else:
        # try Timestamp or Date
        for c in ['Date', 'timestamp', 'published']:
            if c in df.columns:
                df['date_only'] = pd.to_datetime(df[c]).dt.normalize()
                break
    if 'date_only' not in df.columns:
        raise ValueError('No date column found for aggregation')

    df['sent_label'] = df[sentiment_col].map(lambda x: map_sentiment_label(x))
    agg = df.groupby(['date_only','sent_label']).size().unstack(fill_value=0)
    agg = agg[['positive','neutral','negative']] if set(['positive','neutral','negative']).issubset(agg.columns) else agg
    agg = agg.reset_index().rename_axis(None, axis=1)
    return agg


def merge_compare_and_plot(df_orig: pd.DataFrame, title_sent_labels: list, title_col: str, abstract_sent_col: str, date_col: str = 'date'):
    df = df_orig.copy()
    df['finbert_title_sentiment'] = title_sent_labels
    # ensure date_only
    if date_col in df.columns:
        df['date_only'] = pd.to_datetime(df[date_col]).dt.normalize()
    elif 'date_only' not in df.columns:
        raise ValueError('No date column found')

    # aggregate both
    abstract_agg = aggregate_daily(df, abstract_sent_col, date_col)
    title_agg = aggregate_daily(df, 'finbert_title_sentiment', date_col)

    # merge on date_only
    merged = pd.merge(abstract_agg, title_agg, on='date_only', how='outer', suffixes=('_abstract','_title')).fillna(0)

    # reorder columns
    cols = ['date_only']
    for k in ['positive','neutral','negative']:
        cols.append(k + '_abstract' if k + '_abstract' in merged.columns else k + '_x' if k + '_x' in merged.columns else None)
    for k in ['positive','neutral','negative']:
        cols.append(k + '_title' if k + '_title' in merged.columns else k + '_y' if k + '_y' in merged.columns else None)
    cols = [c for c in cols if c is not None and c in merged.columns]
    merged = merged[cols]

    # save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False)

    # Plot: grouped bars for each date
    dates = merged['date_only'].astype(str).tolist()
    x = np.arange(len(dates))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(8, len(dates)*0.6), 6))

    def get_col(name):
        return merged[name].values if name in merged.columns else np.zeros(len(merged))

    ax.bar(x - width*1.5, get_col('positive_abstract'), width, label='abstract_positive')
    ax.bar(x - width*0.5, get_col('neutral_abstract'), width, label='abstract_neutral')
    ax.bar(x + width*0.5, get_col('positive_title'), width, label='title_positive')
    ax.bar(x + width*1.5, get_col('neutral_title'), width, label='title_neutral')

    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha='right')
    ax.set_ylabel('Counts')
    ax.set_title('Daily sentiment counts: abstracts (left) vs titles (right)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG)
    print(f'Plot saved to {OUT_PNG}')
    print(f'Comparison CSV saved to {OUT_CSV}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(DEFAULT_INPUT), help='input CSV with news (titles and abstracts)')
    parser.add_argument('--title-col', default='title')
    parser.add_argument('--abstract-sent-col', default=None, help='existing sentiment column name (auto-detected if not provided)')
    parser.add_argument('--model', default='yiyanghkust/finbert-tone', help='HuggingFace FinBERT model name')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    abstract_col = args.abstract_sent_col or detect_sentiment_column(df)
    if abstract_col is None:
        raise ValueError('Could not detect existing abstract sentiment column; provide with --abstract-sent-col')
    print('Using abstract sentiment column:', abstract_col)

    if args.title_col not in df.columns:
        raise ValueError(f'Title column {args.title_col} not found in input')

    titles = df[args.title_col].fillna('').astype(str)
    print(f'Running FinBERT on {len(titles)} titles (model {args.model})...')
    labels = run_finbert_on_titles(titles, model_name=args.model, batch_size=args.batch_size)

    merge_compare_and_plot(df, labels, args.title_col, abstract_col)


if __name__ == '__main__':
    main()
