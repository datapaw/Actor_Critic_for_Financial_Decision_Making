import pandas as pd
from datetime import datetime

def reformat_csv():
    """Sort news by date/time and convert dates to YYYY-MM-DD format"""
    
    input_csv = 'delete/scraped_news_with_sentiment.csv'
    output_csv = 'delete/scraped_news_sorted.csv'
    
    print("Loading CSV...")
    df = pd.read_csv(input_csv)
    
    print(f"Total articles: {len(df)}")
    print(f"\nOriginal date format samples:")
    print(df['date'].head(3))
    
    # Convert dates to datetime and then to YYYY-MM-DD format
    def parse_date(date_str):
        """Parse various date formats and return YYYY-MM-DD"""
        if pd.isna(date_str) or date_str == '':
            return None
        
        try:
            # Try parsing RFC 2822 format (e.g., "Tue, 09 Dec 2025 12:26:33 GMT")
            dt = pd.to_datetime(date_str, format='mixed')
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            try:
                # Try ISO format (e.g., "2025-12-09T12:02:49Z")
                dt = pd.to_datetime(date_str)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                return date_str  # Keep original if parsing fails
    
    print("\nConverting dates...")
    df['date'] = df['date'].apply(parse_date)
    
    # Create a datetime column for sorting and filtering
    df['datetime_sort'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Sort by date (newest first)
    print("Sorting by date...")
    df_sorted = df.sort_values('datetime_sort', ascending=False, na_position='last')
    
    # Check for duplicates before filtering
    print("\nChecking for duplicates...")
    duplicates = df_sorted[df_sorted.duplicated(subset=['title'], keep=False)]
    if len(duplicates) > 0:
        print(f"⚠ Found {len(duplicates)} duplicate articles (same title)")
        print("\nDuplicate titles:")
        for title in duplicates['title'].unique():
            count = len(duplicates[duplicates['title'] == title])
            print(f"  - {title[:60]}... ({count} copies)")
        
        # Remove duplicates, keeping first occurrence
        df_sorted = df_sorted.drop_duplicates(subset=['title'], keep='first')
        print(f"\n✓ Removed duplicates, {len(df_sorted)} unique articles remaining")
    else:
        print("✓ All articles are unique!")
    
    # Filter articles from 2025-12-08 onwards
    print("\nFiltering articles from 2025-12-08 onwards...")
    cutoff_date = pd.to_datetime('2025-12-08')
    df_filtered = df_sorted[df_sorted['datetime_sort'] >= cutoff_date]
    removed_count = len(df_sorted) - len(df_filtered)
    
    print(f"✓ Removed {removed_count} articles before 2025-12-08")
    print(f"✓ Kept {len(df_filtered)} articles from 2025-12-08 onwards")
    
    # Drop the temporary sorting column
    df_filtered = df_filtered.drop('datetime_sort', axis=1)
    
    # Save to new CSV file
    df_filtered.to_csv(output_csv, index=False)
    
    print(f"\n✓ CSV reformatted, sorted, and filtered!")
    print(f"✓ Saved to: {output_csv}")
    print(f"\nNew date format samples:")
    print(df_filtered['date'].head(5))
    print(f"\nDate range:")
    print(f"  Newest: {df_filtered['date'].iloc[0]}")
    print(f"  Oldest: {df_filtered['date'].iloc[-1]}")
    
    # Check for S&P 500 relevance
    print("\n" + "="*80)
    print("CHECKING S&P 500 / MARKET RELEVANCE")
    print("="*80)
    
    # Top 50 S&P 500 companies and market-related keywords
    sp500_top50 = [
        'Apple', 'Microsoft', 'Amazon', 'NVIDIA', 'Alphabet', 'Google', 'Meta', 'Facebook',
        'Tesla', 'Berkshire', 'Visa', 'JPMorgan', 'Johnson & Johnson', 'Walmart', 'Mastercard',
        'Procter & Gamble', 'Home Depot', 'Bank of America', 'Chevron', 'Pfizer', 'AbbVie',
        'Coca-Cola', 'PepsiCo', 'Costco', 'Netflix', 'Adobe', 'Cisco', 'Merck', 'Intel',
        'Verizon', 'Comcast', 'Thermo Fisher', 'Abbott', 'Disney', 'Nike', 'McDonald',
        'Broadcom', 'Salesforce', 'Oracle', 'Accenture', 'Qualcomm', 'United Health',
        'Bristol-Myers', 'NextEra', 'Texas Instruments', 'Danaher', 'Amgen', 'Wells Fargo',
        'CVS', 'Morgan Stanley', 'Goldman Sachs'
    ]
    
    market_keywords = [
        'S&P 500', 'S&P', 'stock market', 'stocks', 'Fed', 'Federal Reserve', 'interest rate',
        'treasury', 'bond', 'yield', 'inflation', 'GDP', 'employment', 'jobs report',
        'earnings', 'IPO', 'merger', 'acquisition', 'Wall Street', 'Nasdaq', 'Dow Jones',
        'market', 'trading', 'investor', 'equity', 'economic', 'recession', 'growth',
        'monetary policy', 'central bank', 'fiscal', 'oil price', 'commodity', 'currency',
        'dollar', 'euro', 'financial', 'corporate', 'profit', 'revenue', 'valuation'
    ]
    
    def is_sp500_relevant(title, summary):
        """Check if article is related to S&P 500 top companies or market"""
        text = f"{title} {summary}".lower()
        
        # Check for top 50 S&P companies
        for company in sp500_top50:
            if company.lower() in text:
                return True, f"Company: {company}"
        
        # Check for market keywords
        for keyword in market_keywords:
            if keyword.lower() in text:
                return True, f"Keyword: {keyword}"
        
        return False, "No match"
    
    # Analyze relevance
    relevance_results = []
    for idx, row in df_filtered.iterrows():
        is_relevant, reason = is_sp500_relevant(
            str(row.get('title', '')), 
            str(row.get('summary', ''))
        )
        relevance_results.append({
            'title': row.get('title', ''),
            'is_relevant': is_relevant,
            'reason': reason
        })
    
    # Count relevant vs non-relevant
    relevant_count = sum(1 for r in relevance_results if r['is_relevant'])
    non_relevant_count = len(relevance_results) - relevant_count
    
    print(f"\nRelevance Summary:")
    print(f"  ✓ Relevant to S&P 500/Market: {relevant_count} ({relevant_count/len(relevance_results)*100:.1f}%)")
    print(f"  ✗ Not clearly relevant: {non_relevant_count} ({non_relevant_count/len(relevance_results)*100:.1f}%)")
    
    if non_relevant_count > 0:
        print(f"\n⚠ Articles NOT clearly related to S&P 500 top firms or market:")
        count = 0
        for r in relevance_results:
            if not r['is_relevant'] and count < 10:  # Show max 10
                print(f"  - {r['title'][:70]}...")
                count += 1
        if non_relevant_count > 10:
            print(f"  ... and {non_relevant_count - 10} more")
    
    return df_filtered


if __name__ == "__main__":
    df = reformat_csv()
