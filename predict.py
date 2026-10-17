import joblib
import numpy as np
from textblob import TextBlob

# Load trained models
likes_model = joblib.load("likes_prediction_model.pkl")
comments_model = joblib.load("comments_prediction_model.pkl")

def predict_engagement(views, title):
    """Predicts likes and comments for a new video based on views and title."""
    title_length = len(title)
    sentiment_score = TextBlob(title).sentiment.polarity

    # Prepare input features
    features = np.array([[views, title_length, sentiment_score]])

    # Predict likes and comments
    predicted_likes = likes_model.predict(features)[0]
    predicted_comments = comments_model.predict(features)[0]

    return round(predicted_likes), round(predicted_comments)

# Example Usage
if __name__ == "__main__":
    views = float(input("Enter expected views for the new video: "))
    title = input("Enter the video title: ")

    likes, comments = predict_engagement(views, title)
    
    print(f"\n📊 **Predicted Engagement:**")
    print(f"👍 Predicted Likes: {likes}")
    print(f"💬 Predicted Comments: {comments}")
