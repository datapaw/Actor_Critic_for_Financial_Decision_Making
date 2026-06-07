import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')  # too many annoying warnings
import requests
from bs4 import BeautifulSoup
import time

# check if we have FinBERT available
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    print("Heads up: transformers not installed. You'll need it for sentiment analysis.")
    print("Install with: pip install transformers torch")
    FINBERT_AVAILABLE = False

# check for OpenAI (optional - for summaries)
try:
    from openai import OpenAI
    import config
    OPENAI_AVAILABLE = True
except ImportError:
    print("Note: openai package not found (that's fine if you don't need summaries)")
    OPENAI_AVAILABLE = False


class FinancialNewsWebScraper:
    # scrapes news from various financial sites
    
    def __init__(self, start_date: str = None, end_date: str = None):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        # date filtering
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        
        # setup FinBERT for sentiment if available
        self.finbert_available = FINBERT_AVAILABLE
        if self.finbert_available:
            print("Loading FinBERT... (this takes a sec)")
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            print(f"FinBERT ready on {self.device}")
        else:
            print("Skipping sentiment analysis (FinBERT not available)")
        
        # setup OpenAI if we have the key
        self.openai_available = OPENAI_AVAILABLE
        if self.openai_available and hasattr(config, 'OPENAI_API_KEY'):
            self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            print("OpenAI client ready for summaries")
        else:
            self.openai_available = False
            print("OpenAI not available - summaries will not be generated")
    
    def is_in_date_range(self, date_str: str) -> bool:
        """Check if article date is within the specified range"""
        if not date_str:
            return True  # Include articles without dates
        
        if not self.start_date and not self.end_date:
            return True  # No filtering
        
        try:
            # Parse various date formats
            article_date = pd.to_datetime(date_str)
            
            if self.start_date and article_date < self.start_date:
                return False
            if self.end_date and article_date > self.end_date:
                return False
            
            return True
        except:
            return True  # Include if we can't parse the date
    
    def generate_summary(self, title: str, url: str) -> str:
        """Generate summary using OpenAI when summary is missing"""
        if not self.openai_available or not title:
            return ""
        
        try:
            # Try to fetch and extract content from the article URL
            article_content = ""
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Try to extract article text from common tags
                paragraphs = soup.find_all('p', limit=5)  # Get first 5 paragraphs
                article_content = ' '.join([p.get_text(strip=True) for p in paragraphs])
                
                # Limit content length to avoid token limits
                article_content = article_content[:1000]
            except:
                pass  # If fetching fails, we'll just use the title
            
            # Create prompt based on available information
            if article_content:
                prompt = f"Title: {title}\n\nArticle excerpt: {article_content}\n\nProvide a brief 2-3 sentence summary focusing on the key financial implications and market impact."
            else:
                prompt = f"Based on this financial news headline, provide a brief 2-3 sentence summary of what the article likely covers, focusing on potential financial implications: {title}"
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial news analyst. Create concise, informative summaries that highlight key financial implications and market relevance."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            print(f"  Error generating summary with OpenAI: {e}")
            return ""
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using FinBERT"""
        if not self.finbert_available or not text:
            return {
                'sentiment': 'neutral',
                'positive_score': 0.0,
                'negative_score': 0.0,
                'neutral_score': 0.0
            }
        
        try:
            # Tokenize and prepare input (max 512 tokens)
            inputs = self.tokenizer(text, return_tensors="pt", 
                                   truncation=True, max_length=512, 
                                   padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT outputs: [positive, negative, neutral]
            scores = predictions[0].cpu().numpy()
            
            sentiment_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
            sentiment = sentiment_map[scores.argmax()]
            
            return {
                'sentiment': sentiment,
                'positive_score': float(scores[0]),
                'negative_score': float(scores[1]),
                'neutral_score': float(scores[2])
            }
        except Exception as e:
            print(f"  Error analyzing sentiment: {e}")
            return {
                'sentiment': 'neutral',
                'positive_score': 0.0,
                'negative_score': 0.0,
                'neutral_score': 0.0
            }
    
    def scrape_bloomberg_market_news(self) -> List[Dict]:
        """Scrape market news from Bloomberg via RSS"""
        articles = []
        
        try:
            print("Scraping Bloomberg Markets...")
            url = 'https://feeds.bloomberg.com/markets/news.rss'
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'xml')
            
            for item in soup.find_all('item', limit=500):
                title_tag = item.find('title')
                desc_tag = item.find('description')
                link_tag = item.find('link')
                date_tag = item.find('pubDate')
                
                if title_tag and link_tag:
                    date_str = date_tag.get_text(strip=True) if date_tag else None
                    
                    # Apply date filter
                    if not self.is_in_date_range(date_str):
                        continue
                    
                    articles.append({
                        'source': 'Bloomberg',
                        'title': title_tag.get_text(strip=True),
                        'link': link_tag.get_text(strip=True),
                        'summary': desc_tag.get_text(strip=True)[:500] if desc_tag else '',
                        'date': date_str
                    })
            
            print(f"  Found {len(articles)} articles from Bloomberg")
        
        except Exception as e:
            print(f"  Error scraping Bloomberg: {e}")
        
        return articles
    
    def scrape_seeking_alpha_news(self) -> List[Dict]:
        """Scrape market news from Seeking Alpha RSS"""
        articles = []
        
        try:
            print("Scraping Seeking Alpha...")
            url = 'https://seekingalpha.com/feed.xml'
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'xml')
            
            for item in soup.find_all('item', limit=500):
                title_tag = item.find('title')
                desc_tag = item.find('description')
                link_tag = item.find('link')
                date_tag = item.find('pubDate')
                
                if title_tag and link_tag:
                    date_str = date_tag.get_text(strip=True) if date_tag else None
                    
                    # Apply date filter
                    if not self.is_in_date_range(date_str):
                        continue
                    
                    articles.append({
                        'source': 'Seeking Alpha',
                        'title': title_tag.get_text(strip=True),
                        'link': link_tag.get_text(strip=True),
                        'summary': desc_tag.get_text(strip=True)[:500] if desc_tag else '',
                        'date': date_str
                    })
            
            print(f"  Found {len(articles)} articles from Seeking Alpha")
        
        except Exception as e:
            print(f"  Error scraping Seeking Alpha: {e}")
        
        return articles
    
    def scrape_marketwatch_news(self) -> List[Dict]:
        """Scrape news from MarketWatch RSS"""
        articles = []
        
        try:
            print("Scraping MarketWatch...")
            url = 'https://feeds.marketwatch.com/marketwatch/topstories/'
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'xml')
            
            for item in soup.find_all('item', limit=500):
                title_tag = item.find('title')
                desc_tag = item.find('description')
                link_tag = item.find('link')
                date_tag = item.find('pubDate')
                
                if title_tag and link_tag:
                    date_str = date_tag.get_text(strip=True) if date_tag else None
                    
                    # Apply date filter
                    if not self.is_in_date_range(date_str):
                        continue
                    
                    articles.append({
                        'source': 'MarketWatch',
                        'title': title_tag.get_text(strip=True),
                        'link': link_tag.get_text(strip=True),
                        'summary': desc_tag.get_text(strip=True)[:500] if desc_tag else '',
                        'date': date_str
                    })
            
            print(f"  Found {len(articles)} articles from MarketWatch")
        
        except Exception as e:
            print(f"  Error scraping MarketWatch: {e}")
        
        return articles
    
    def scrape_reuters_markets(self) -> List[Dict]:
        """Scrape news from Reuters Markets RSS"""
        articles = []
        
        try:
            print("Scraping Reuters Markets...")
            url = 'https://www.reutersagency.com/feed/?taxonomy=best-topics&output=rss'
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'xml')
            
            for item in soup.find_all('item', limit=500):
                title_tag = item.find('title')
                desc_tag = item.find('description')
                link_tag = item.find('link')
                date_tag = item.find('pubDate')
                
                if title_tag and link_tag:
                    date_str = date_tag.get_text(strip=True) if date_tag else None
                    
                    # Apply date filter
                    if not self.is_in_date_range(date_str):
                        continue
                    
                    articles.append({
                        'source': 'Reuters',
                        'title': title_tag.get_text(strip=True),
                        'link': link_tag.get_text(strip=True),
                        'summary': desc_tag.get_text(strip=True)[:500] if desc_tag else '',
                        'date': date_str
                    })
            
            print(f"  Found {len(articles)} articles from Reuters")
        
        except Exception as e:
            print(f"  Error scraping Reuters: {e}")
        
        return articles


    def scrape_yahoo_finance_rss(self) -> List[Dict]:
        """Scrape news from Yahoo Finance RSS"""
        articles = []
        
        try:
            print("Scraping Yahoo Finance...")
            url = 'https://finance.yahoo.com/news/rssindex'
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'xml')
            
            for item in soup.find_all('item', limit=500):
                title_tag = item.find('title')
                desc_tag = item.find('description')
                link_tag = item.find('link')
                date_tag = item.find('pubDate')
                
                if title_tag and link_tag:
                    date_str = date_tag.get_text(strip=True) if date_tag else None
                    
                    # Apply date filter
                    if not self.is_in_date_range(date_str):
                        continue
                    
                    articles.append({
                        'source': 'Yahoo Finance',
                        'title': title_tag.get_text(strip=True),
                        'link': link_tag.get_text(strip=True),
                        'summary': desc_tag.get_text(strip=True)[:500] if desc_tag else '',
                        'date': date_str
                    })
            
            print(f"  Found {len(articles)} articles from Yahoo Finance")
        
        except Exception as e:
            print(f"  Error scraping Yahoo Finance: {e}")
        
        return articles

    def scrape_all_sources(self) -> List[Dict]:
        """Scrape news from all sources"""
        all_articles = []
        
        print("Starting web scraping from multiple financial news sources...\n")
        
        # Scrape all sources
        all_articles.extend(self.scrape_bloomberg_market_news())
        time.sleep(1)  # Be respectful with requests
        
        all_articles.extend(self.scrape_seeking_alpha_news())
        time.sleep(1)
        
        all_articles.extend(self.scrape_marketwatch_news())
        time.sleep(1)
        
        all_articles.extend(self.scrape_reuters_markets())
        time.sleep(1)
        
        all_articles.extend(self.scrape_yahoo_finance_rss())
        time.sleep(1)
        
        print(f"\nTotal articles scraped: {len(all_articles)}")
        
        # Remove duplicates by title
        unique_articles = {}
        for article in all_articles:
            title = article.get('title', '')
            if title and title not in unique_articles:
                unique_articles[title] = article
        
        all_articles = list(unique_articles.values())
        print(f"Unique articles: {len(all_articles)}")
        
        # Generate summaries for articles without them
        if self.openai_available and all_articles:
            articles_without_summary = [a for a in all_articles if not a.get('summary') or a.get('summary').strip() == '']
            if articles_without_summary:
                print(f"\nGenerating summaries for {len(articles_without_summary)} articles using OpenAI...")
                for i, article in enumerate(articles_without_summary, 1):
                    summary = self.generate_summary(article.get('title', ''), article.get('link', ''))
                    if summary:
                        article['summary'] = summary
                    
                    if i % 10 == 0:
                        print(f"  Generated {i}/{len(articles_without_summary)} summaries...")
                        time.sleep(1)  # Rate limiting
                
                print(f"  Summary generation complete!")
        
        # Analyze sentiment for each article
        if self.finbert_available and all_articles:
            print("\nAnalyzing sentiment with FinBERT...")
            for i, article in enumerate(all_articles, 1):
                # Use summary if available, otherwise use title
                # Combine title and summary for best results when both available
                text_for_analysis = ""
                if article.get('title'):
                    text_for_analysis = article['title']
                if article.get('summary'):
                    text_for_analysis += " " + article['summary']
                
                sentiment_result = self.analyze_sentiment(text_for_analysis.strip())
                article.update(sentiment_result)
                
                if i % 20 == 0:
                    print(f"  Processed {i}/{len(all_articles)} articles...")
            
            print(f"  Sentiment analysis complete!")
        
        return all_articles
    
    def print_summary(self, articles: List[Dict]):
        """Print summary of scraped articles"""
        if not articles:
            print("No articles to display")
            return
        
        print("\n" + "="*80)
        print("SCRAPED NEWS SUMMARY")
        print("="*80)
        
        print(f"\nTotal articles: {len(articles)}")
        
        print(f"\nArticles by source:")
        sources = {}
        for a in articles:
            src = a.get('source', 'Unknown')
            sources[src] = sources.get(src, 0) + 1
        
        for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"  {src}: {count}")
        
        # Sentiment distribution
        if any('sentiment' in a for a in articles):
            print(f"\nSentiment distribution:")
            sentiments = {}
            for a in articles:
                sent = a.get('sentiment', 'unknown')
                sentiments[sent] = sentiments.get(sent, 0) + 1
            
            for sent, count in sorted(sentiments.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(articles)) * 100
                print(f"  {sent.capitalize()}: {count} ({pct:.1f}%)")
        
        print(f"\nArticles by date (last 30 days):")
        dates = {}
        for a in articles:
            date_str = a.get('date', 'Unknown')
            if date_str:
                # Try to extract just the date part
                date_part = date_str.split('T')[0] if 'T' in date_str else date_str[:10]
                dates[date_part] = dates.get(date_part, 0) + 1
        
        if dates:
            for date, count in sorted(dates.items(), reverse=True)[:30]:
                print(f"  {date}: {count}")
        else:
            print("  No date information available")
        
        print("\n" + "="*80)
        print("SAMPLE ARTICLES BY SOURCE (2 from each)")
        print("="*80)
        
        # Group articles by source
        by_source = {}
        for article in articles:
            source = article.get('source', 'Unknown')
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(article)
        
        # Display 2 from each source
        for source in ['Bloomberg', 'Seeking Alpha', 'MarketWatch', 'Reuters', 'Yahoo Finance']:
            if source in by_source:
                print(f"\n--- {source.upper()} ---")
                for i, article in enumerate(by_source[source][:2], 1):
                    print(f"\n{i}. Title: {article.get('title')}")
                    if article.get('date'):
                        print(f"   Date: {article.get('date')}")
                    if article.get('sentiment'):
                        sent = article.get('sentiment', 'neutral')
                        pos = article.get('positive_score', 0)
                        neg = article.get('negative_score', 0)
                        print(f"   Sentiment: {sent.upper()} (pos: {pos:.3f}, neg: {neg:.3f})")
                    print(f"   Link: {article.get('link')[:80]}...")
                    if article.get('summary'):
                        print(f"   Summary: {article.get('summary')[:150]}...")
    
    def save_articles(self, articles: List[Dict], output_path: str = 'delete/scraped_news_with_sentiment.csv'):
        """Save scraped articles with sentiment to CSV"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if articles:
            df = pd.DataFrame(articles)
            
            # Reorder columns for better readability
            column_order = ['date', 'source', 'title', 'sentiment', 
                          'positive_score', 'negative_score', 'neutral_score',
                          'summary', 'link']
            
            # Only include columns that exist
            existing_cols = [col for col in column_order if col in df.columns]
            other_cols = [col for col in df.columns if col not in existing_cols]
            df = df[existing_cols + other_cols]
            
            df.to_csv(output_path, index=False)
            print(f"\nArticles with sentiment saved to {output_path}")
            print(f"Total records: {len(df)}")
            
            # Summary statistics
            if 'sentiment' in df.columns:
                print(f"\nSentiment summary:")
                print(f"  Positive: {(df['sentiment'] == 'positive').sum()}")
                print(f"  Negative: {(df['sentiment'] == 'negative').sum()}")
                print(f"  Neutral: {(df['sentiment'] == 'neutral').sum()}")
                
                if 'positive_score' in df.columns:
                    print(f"\nAverage sentiment scores:")
                    print(f"  Positive: {df['positive_score'].mean():.3f}")
                    print(f"  Negative: {df['negative_score'].mean():.3f}")
                    print(f"  Neutral: {df['neutral_score'].mean():.3f}")
            
            return df
        
        return None


    def load_existing_articles(self, csv_path: str) -> set:
        """Load existing article titles from CSV to avoid duplicates"""
        try:
            if Path(csv_path).exists():
                df = pd.read_csv(csv_path)
                existing_titles = set(df['title'].tolist())
                print(f"Loaded {len(existing_titles)} existing articles from {csv_path}")
                return existing_titles
            else:
                print(f"No existing CSV found at {csv_path}, starting fresh")
                return set()
        except Exception as e:
            print(f"Error loading existing articles: {e}")
            return set()
    
    def append_new_articles(self, new_articles: List[Dict], csv_path: str):
        """Append only new articles to existing CSV"""
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        
        if not new_articles:
            print("No new articles to append")
            return None
        
        new_df = pd.DataFrame(new_articles)
        
        # Reorder columns for better readability
        column_order = ['date', 'source', 'title', 'sentiment', 
                      'positive_score', 'negative_score', 'neutral_score',
                      'summary', 'link']
        
        existing_cols = [col for col in column_order if col in new_df.columns]
        other_cols = [col for col in new_df.columns if col not in existing_cols]
        new_df = new_df[existing_cols + other_cols]
        
        # Append to existing CSV or create new one
        if Path(csv_path).exists():
            new_df.to_csv(csv_path, mode='a', header=False, index=False)
            print(f"\nAppended {len(new_df)} new articles to {csv_path}")
        else:
            new_df.to_csv(csv_path, index=False)
            print(f"\nCreated new CSV with {len(new_df)} articles at {csv_path}")
        
        return new_df


def scrape_and_update():
    """Single scraping run - check for new articles and update CSV"""
    
    print("="*80)
    print(f"Financial News Scraper - Update Run")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    CSV_PATH = 'delete/scraped_news_with_sentiment.csv'
    START_DATE = '2024-01-01'
    END_DATE = '2025-12-31'
    
    scraper = FinancialNewsWebScraper(start_date=START_DATE, end_date=END_DATE)
    
    # Load existing articles
    existing_titles = scraper.load_existing_articles(CSV_PATH)
    
    # Scrape all sources
    print("\nScraping news sources...")
    all_articles = []
    all_articles.extend(scraper.scrape_bloomberg_market_news())
    time.sleep(1)
    all_articles.extend(scraper.scrape_seeking_alpha_news())
    time.sleep(1)
    all_articles.extend(scraper.scrape_marketwatch_news())
    time.sleep(1)
    all_articles.extend(scraper.scrape_reuters_markets())
    time.sleep(1)
    all_articles.extend(scraper.scrape_yahoo_finance_rss())
    
    print(f"\nTotal articles scraped: {len(all_articles)}")
    
    # Filter only new articles
    new_articles = [a for a in all_articles if a.get('title') not in existing_titles]
    print(f"New articles found: {len(new_articles)}")
    
    if new_articles:
        # Generate summaries for articles without them
        if scraper.openai_available:
            articles_needing_summary = [a for a in new_articles if not a.get('summary') or a.get('summary').strip() == '']
            if articles_needing_summary:
                print(f"\nGenerating summaries for {len(articles_needing_summary)} articles...")
                for i, article in enumerate(articles_needing_summary, 1):
                    summary = scraper.generate_summary(article.get('title', ''), article.get('link', ''))
                    if summary:
                        article['summary'] = summary
                    if i % 5 == 0:
                        print(f"  Generated {i}/{len(articles_needing_summary)} summaries...")
                        time.sleep(1)
        
        # Analyze sentiment
        if scraper.finbert_available:
            print(f"\nAnalyzing sentiment for {len(new_articles)} new articles...")
            for i, article in enumerate(new_articles, 1):
                text_for_analysis = ""
                if article.get('title'):
                    text_for_analysis = article['title']
                if article.get('summary'):
                    text_for_analysis += " " + article['summary']
                
                sentiment_result = scraper.analyze_sentiment(text_for_analysis.strip())
                article.update(sentiment_result)
                
                if i % 10 == 0:
                    print(f"  Processed {i}/{len(new_articles)} articles...")
        
        # Append to CSV
        scraper.append_new_articles(new_articles, CSV_PATH)
        
        # Print summary of new articles
        print("\n" + "="*80)
        print("NEW ARTICLES SUMMARY")
        print("="*80)
        print(f"\nTotal new articles: {len(new_articles)}")
        
        sources = {}
        for a in new_articles:
            src = a.get('source', 'Unknown')
            sources[src] = sources.get(src, 0) + 1
        
        print("\nBy source:")
        for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"  {src}: {count}")
        
        if any('sentiment' in a for a in new_articles):
            sentiments = {}
            for a in new_articles:
                sent = a.get('sentiment', 'unknown')
                sentiments[sent] = sentiments.get(sent, 0) + 1
            
            print("\nSentiment:")
            for sent, count in sorted(sentiments.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(new_articles)) * 100
                print(f"  {sent.capitalize()}: {count} ({pct:.1f}%)")
    else:
        print("\nNo new articles to process")
    
    print("\n" + "="*80)


def run_scheduler():
    """Run scraper on schedule: every 30 minutes"""
    import schedule
    
    print("="*80)
    print("Financial News Scraper - Scheduler Mode")
    print("="*80)
    print("\nScheduled runs every 30 minutes")
    print("Press Ctrl+C to stop\n")
    
    # Schedule job to run every 30 minutes
    schedule.every(30).minutes.do(scrape_and_update)
    
    # Run immediately on start
    print("Running initial scrape...")
    scrape_and_update()
    
    # Keep running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\n\nScheduler stopped by user")
        return


def main():
    """Main execution"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--schedule':
        # Run in scheduled mode
        try:
            import schedule
        except ImportError:
            print("Error: 'schedule' library not installed")
            print("Install with: pip install schedule")
            return
        run_scheduler()
    else:
        # Run once
        scrape_and_update()


if __name__ == "__main__":
    main()
