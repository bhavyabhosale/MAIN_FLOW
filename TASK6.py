# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
file_path = '/mnt/data/disney_plus_titles.csv'
data = pd.read_csv(file_path)

# Display the first few rows
print(data.head())

# Check data information
print(data.info())

### TASK 1: TIME SERIES ANALYSIS ###

# Convert 'release_date' or equivalent column to datetime
if 'release_date' in data.columns:
    data['release_date'] = pd.to_datetime(data['release_date'])
    data.set_index('release_date', inplace=True)

    # Resample data to count titles per month
    monthly_counts = data.resample('M').size()

    # Plot the time series data
    plt.figure(figsize=(14, 7))
    plt.plot(monthly_counts, marker='o')
    plt.title('Monthly Content Releases')
    plt.xlabel('Date')
    plt.ylabel('Number of Releases')
    plt.grid(True)
    plt.show()

    # Decompose the time series to check trend and seasonality
    decomposition = seasonal_decompose(monthly_counts, model='additive')
    decomposition.plot()
    plt.show()

    # ARIMA Model for Forecasting
    model = ARIMA(monthly_counts, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=12)
    print(forecast)

### TASK 2: SENTIMENT ANALYSIS OR TEXT MINING ###

# Check if a description or other text field exists
if 'description' in data.columns:
    # Function to calculate sentiment score
    def get_sentiment(text):
        return TextBlob(str(text)).sentiment.polarity

    # Apply sentiment analysis
    data['sentiment_score'] = data['description'].apply(get_sentiment)

    # Plot sentiment scores
    plt.figure(figsize=(10, 6))
    sns.histplot(data['sentiment_score'], kde=True, color='purple')
    plt.title('Sentiment Distribution of Descriptions')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()

### TASK 3: CLUSTERING OR CLASSIFICATION ###

# Select numeric features for clustering (ensure relevant columns are used)
features = ['rating', 'runtime', 'year']  # Adjust based on available columns
if all(col in data.columns for col in features):
    # Clean data and handle missing values
    clustering_data = data[features].dropna()

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(pca_data)

    # Add cluster information to the data
    clustering_data['Cluster'] = clusters

    # Plot Clustering Results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters, palette='viridis')
    plt.title('K-Means Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
else:
    print("Required features for clustering not found.")
