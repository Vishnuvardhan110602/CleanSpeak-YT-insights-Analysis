import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

# Load the dataset
df = pd.read_csv(r"C:\Users\ganes\Documents\UCYC6Vcczj8v-Y5OgpEJTFBw_videos.xls.csv")

# Feature Engineering: Extract new features
df['Title_Length'] = df['Title'].apply(lambda x: len(str(x)))
df['Sentiment_Score'] = df['Title'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Engagement Metrics (Likes per View, Comments per View)
df['Likes_Per_View'] = df['Likes'] / df['Views']
df['Comments_Per_View'] = df['Comments'] / df['Views']

# Drop missing values (important for ratio-based columns)
df = df.dropna()

# Features and Targets
X = df[['Views', 'Title_Length', 'Sentiment_Score']]
y_likes = df['Likes']
y_comments = df['Comments']

# Train-test split
X_train, X_test, y_likes_train, y_likes_test = train_test_split(X, y_likes, test_size=0.2, random_state=42)
X_train, X_test, y_comments_train, y_comments_test = train_test_split(X, y_comments, test_size=0.2, random_state=42)

# Define ML Pipelines
likes_model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

comments_model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the models
likes_model.fit(X_train, y_likes_train)
comments_model.fit(X_train, y_comments_train)

# Save the trained models
joblib.dump(likes_model, 'likes_prediction_model.pkl')
joblib.dump(comments_model, 'comments_prediction_model.pkl')

print("✅ Model training complete! Saved models: likes_prediction_model.pkl, comments_prediction_model.pkl")
