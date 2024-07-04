

import requests
import json
import jsonlines
from datetime import datetime, timedelta

API_KEY = 'YOUR_API_KEY'
API_ENDPOINT = 'https://www.alphavantage.co/query'

def fetch_news(api_key, keyword, topic, time_from, time_to, page_size=100):
    """
    Fetch news articles from Alpha Vantage API related to a specific keyword and topic within a date range.

    Args:
        api_key (str): API key for authentication.
        keyword (str): Keyword to search for.
        topic (str): Topic to search under.
       time_from (str): Start time in 'YYYYMMDDTHHMM' format.
        time_to (str): End time in 'YYYYMMDDTHHMM' format.
        page_size (int): Number of articles to fetch per request.

    Returns:
        articles (list): List of dictionaries containing article information.
    """
    params = {
        'function': 'NEWS_SENTIMENT',
        'apikey': api_key,
        'topics': topic,
        'q': keyword,
        'time_from': time_from,
        'time_to': time_to,
        'limit': page_size,
    }

    try:
        response = requests.get(API_ENDPOINT, params=params)
        response.raise_for_status()  # Raise an exception for bad responses
        data = response.json()
        articles = data.get('feed', [])
        return articles
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return []

def save_to_jsonl(data, filename):
    """
    Save data to a JSONL file.

    Args:
        data (list): Data to save.
        filename (str): Name of the JSONL file.
    """
    existing_data = []

    # Check if the file exists and read existing data
    try:
        with jsonlines.open(filename, mode='r') as reader:
            existing_data = list(reader.iter(type=dict))
    except FileNotFoundError:
        pass  # File doesn't exist, proceed to create a new one

    # Check each article and append only if it's not already present
    new_data = [article for article in data if article not in existing_data]

    if new_data:
        with jsonlines.open(filename, mode='a') as writer:
            writer.write_all(new_data)
            print(f"Appended {len(new_data)} new articles to {filename}.")
    else:
        print("No new articles appended.")

def main():
    
    time_from = '20230510T0130'  # Start time in YYYYMMDDTHHMM format
    # t1= '202301210T0130'  
    time_to = datetime.now().strftime('%Y%m%dT%H%M')  # End time in YYYYMMDDTHHMM format

    # Fetch news articles related to NFT under the topic blockchain within the last year
    articles = fetch_news(API_KEY, 'NFT', 'blockchain', time_from, time_to)

    # Process and append fetched data to news.jsonl
    if articles:
        print(f"Fetched {len(articles)} articles from the API.")
        save_to_jsonl(articles, 'news.jsonl')
    else:
        print("No articles fetched from the API.")

if __name__ == '__main__':
    main()

