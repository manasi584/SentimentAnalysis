import matplotlib.pyplot as plt
import pandas as pd
import re
import jsonlines
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.dates as mdates
from scipy import stats
from scipy.signal import argrelextrema
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

JSON_FILE = "news.jsonl"


def read_jsonl(file_path):
    """Reads a jsonl file and returns it as a list of dictionaries.
    
    Args:
        file_path (str): The path to the jsonl file.

    Returns:
        data (list): A list of dictionaries each containing a single line from the jsonl file.
    """
    with jsonlines.open(file_path) as file:
        data = [article for article in file]
    return data


def find_titles(data):
    """Finds and returns all the titles containing a specific keyword in the data.
    
    Args:
        data (list): The data from which to extract the titles.
        

    Returns:
        titles (list): A list of titles containing the keyword.
    """
    titles = []
    for entry in data:
        title = entry['title']
        titles.append(title)
    return titles


def process_words(titles):
    """Processes the titles by removing newlines, non-alphanumeric characters and non-alpha words,
    and converting to lowercase.
    
    Args:
        titles (list): A list of titles.

    Returns:
        processed_titles (list): A list of processed titles.
    """
    processed_titles = []
    for title in titles:
        title = title.replace('\n', ' ')
        title = title.strip()
        title = re.sub(r'[^\w\s;]', '', title)
        title = title.lower()
        title = ' '.join(word for word in title.split() if word.isalpha())
        processed_titles.append(title)
    return processed_titles


def sentiment(title):
    """
    Compute the sentiment score of a given text.

    Args:
        title (str): Text to compute sentiment score for.

    Returns:
        sentiment (float): Sentiment score of the text.
    """

    # Calculate Sentiment Score
    blob = TextBlob(str(title))
    sentiment = blob.sentiment.polarity
    return sentiment


def remove_stop_words(processed_titles, stop_words_file):
    """Removes stop words from the titles.
    
    Args:
        processed_titles (list): A list of processed titles.
        stop_words_file (str): The path to the file containing the stop words.

    Returns:
        filtered_titles (list): A list of titles with stop words removed.
    """
    with open(stop_words_file, 'r') as file:
        stop_words = set(file.read().lower().split())

    filtered_titles = []
    for title in processed_titles:
        title_words = title.split()  # Split the title into words
        filtered_title = ' '.join(word for word in title_words if word.lower() not in stop_words)  # Iterate over words, not characters
        filtered_titles.append(filtered_title)
    return filtered_titles


def find_datetimes(data):
    """Finds and returns all the publication dates in the data.
    
    Args:
        data (list): The data from which to extract the publication dates.

    Returns:
        date_times (list): A list of publication dates.
    """
    date_times = [entry['time_published'] for entry in data]
    return date_times


def monthly_sentiment(sentiment_scores, timestamps):
    """
    Calculates the monthly average sentiment scores.

    Args:
        sentiment_scores (list): List of sentiment scores.
        timestamps (list): List of corresponding timestamps.

    Returns:
        pandas.DataFrame: DataFrame with monthly average sentiment scores.
    """
    timestamps = pd.to_datetime(timestamps, format='%Y%m%dT%H%M%S')

    # Ensure timestamps are at the end of each month
    timestamps = timestamps.to_period('M').to_timestamp('M')

    df = pd.DataFrame({'Sentiment': sentiment_scores, 'Timestamp': timestamps})
    
    monthly_sentiment = df.groupby(pd.Grouper(key='Timestamp', freq='ME')).mean()
    
    return monthly_sentiment

def categorize_sentiment_scores(sentiment_scores):
    """
    Categorize sentiment scores into 'Very bad', 'Bad', 'Good' and 'Very good'.

    Args:
        sentiment_scores (list): List of sentiment scores.

    Returns:
        categories (list): List of categories corresponding to the sentiment scores.
    """
    # Instantiate the categories empty list
    categories = []
    for score in sentiment_scores:
        if -1.0 <= score < -0.5:
            categories.append('Very bad')
        elif -0.5 <= score < 0:
            categories.append('Bad')
        elif 0 < score <= 0.5:
            categories.append('Good')
        else:
            categories.append('Very good')
    return categories

def plot_sentiment_histogram(categories):
    """
    Plots a histogram of sentiment categories.

    Args:
        categories (list): List of sentiment categories ('Very bad', 'Bad', 'Good', 'Very good').
    """

    category_counts = {
        'Very bad': categories.count('Very bad'),
        'Bad': categories.count('Bad'),
        'Good': categories.count('Good'),
        'Very good': categories.count('Very good')
    }


    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(category_counts.keys()), y=list(category_counts.values()), palette='viridis')


    plt.title('Sentiment Category Distribution', fontsize=16, pad=20)
    plt.xlabel('Sentiment Category', fontsize=13, labelpad=15)
    plt.ylabel('Count', fontsize=13, labelpad=15)

    
    plt.tight_layout()
    plt.show()

def create_wordclouds(titles,categories):
    """
    Creates and displays word clouds for each category of sentiment.

    Args:
        df (DataFrame): A DataFrame containing the titles and their associated sentiment categories.

    Returns:
        Word clouds for each sentiment category.
    """
    df = pd.DataFrame({
    'Title': titles,
    'Category': categories
})
    for category in ['Very bad', 'Bad', 'Good', 'Very good']:
        # Filter summaries of a specific category
        summaries = df[df['Category'] == category]['Title']


        if summaries.empty:
            print(f"No titles in category {category}")
            continue


        text = ' '.join(summaries)

        wordcloud = WordCloud(width=1400, height=1000).generate(text)


        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud for {category} Sentiment')
        plt.axis("off")
        plt.show()


def plot_volume_over_time(dates):
    """
    Plots the number of articles over time.

    Args:
        dates (list): A list of dates when the articles were published.
    
    Returns:
        A matplotlib line plot of the number of articles over time.
    """
    
    df = pd.DataFrame({'Timestamp': pd.to_datetime(dates,format='%Y%m%dT%H%M%S')})
    df = df.sort_values('Timestamp')  # Ensure data is sorted by date
    df['Count'] = 1


    df.set_index('Timestamp', inplace=True)
    df.resample('W').sum().plot(kind='line')  # Resample by week
    plt.title('Number of Articles About NFTs Over Time')
    plt.ylabel('Count')
    plt.show()




def plot_sentiment_over_time_violin(sentiment_scores, dates):
    """
    Plots sentiment scores over time using a violin plot.

    Args:
        sentiment_scores (list): A list of sentiment scores.
        dates (list): A list of dates corresponding to the sentiment scores.
    """

    df = pd.DataFrame({'Sentiment': sentiment_scores, 'Timestamp': pd.to_datetime(dates)})
    df = df.sort_values('Timestamp')  # Ensure data is sorted by date


    plt.figure(figsize=(12, 7))
    sns.violinplot(x=df['Timestamp'].dt.date, y=df['Sentiment'], inner='quartile')

  
    plt.title('Sentiment Scores Over Time (Violin Plot)', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=13, labelpad=15)
    plt.ylabel('Sentiment Score', fontsize=13, labelpad=15)

    plt.xticks(rotation=45)

    
    plt.tight_layout()
    plt.show()






def main():
    # Read the latest JSON file with all news data
    data = read_jsonl(JSON_FILE)
    
    # Find titles 
    titles = find_titles(data)
    

    # Process titles and remove stopwords
    processed_titles = process_words(titles)
    filtered_titles = remove_stop_words(processed_titles, 'stop_words_english.txt')

    

    # Get dates of articles
    dates = find_datetimes(data)

    # Calculate sentiment scores for filtered titles
    polarities = [sentiment(title) for title in filtered_titles]

    categories=categorize_sentiment_scores(polarities)


    create_wordclouds(titles,categories)

    plot_sentiment_histogram(categories)
  
    

    # Calculate monthly average sentiment
    monthly_sentiments = monthly_sentiment(polarities, dates)
    
  
    # Plot volume of articles over time
    plot_volume_over_time(dates)

    

    plot_sentiment_over_time_violin(polarities, dates)
    


if __name__ == '__main__':
    main()
