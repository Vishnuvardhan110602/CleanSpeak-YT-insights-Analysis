



# -------------------------------------------------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import emoji
from collections import Counter
import plotly.express as px
from bs4 import BeautifulSoup
import joblib  # Import joblib to load model

# Function to clean HTML and preserve YouTube links
def clean_html(text):
    soup = BeautifulSoup(text, "html.parser")
    for a_tag in soup.find_all("a"):  # Find all links
        href = a_tag.get("href", "")  # Extract href link
        if href:
            a_tag.replace_with(f"{a_tag.text} ({href})")  # Replace with visible text + URL
    return soup.get_text()

# Function for Comments Analysis
def analyze_comments(comments_file):
    comments = pd.read_csv(comments_file)
    comments.dropna(inplace=True)

    # Clean HTML in comments
    comments['Comment'] = comments['Comment'].apply(lambda x: clean_html(x) if isinstance(x, str) else x)

    # Sentiment Analysis
    comments['polarity'] = [TextBlob(comment).sentiment.polarity if isinstance(comment, str) else 0 for comment in comments['Comment']]

    analysis_type = st.sidebar.radio("Choose Analysis", ["Sentiment Analysis", "WordCloud", "Emoji Analysis"])

    if analysis_type == "Sentiment Analysis":
        sentiment_choice = st.radio("Select Sentiment Type", ["Overall", "Positive", "Negative", "Neutral"])

        if sentiment_choice == "Overall":
            st.subheader("Sentiment Score Distribution")
            fig = px.histogram(comments, x='polarity', nbins=50, title="Sentiment Score Distribution", color_discrete_sequence=['#FF6361'])
            st.plotly_chart(fig, use_container_width=True)

        elif sentiment_choice == "Positive":
            positive_comments = comments[comments['polarity'] > 0]
            st.write(f"### Positive Comments (Total: {len(positive_comments)})")
            st.dataframe(positive_comments[['Comment', 'polarity']].head(10))

        elif sentiment_choice == "Negative":
            negative_comments = comments[comments['polarity'] < 0]
            st.write(f"### Negative Comments (Total: {len(negative_comments)})")
            st.dataframe(negative_comments[['Comment', 'polarity']].head(10))

        elif sentiment_choice == "Neutral":
            neutral_comments = comments[comments['polarity'] == 0]
            st.write(f"### Neutral Comments (Total: {len(neutral_comments)})")
            st.dataframe(neutral_comments[['Comment', 'polarity']].head(10))

    elif analysis_type == "WordCloud":
        wordcloud_type = st.radio("Choose Sentiment for WordCloud", ["Positive", "Negative"])
        st.subheader(f"WordCloud for {wordcloud_type} Comments")

        wordcloud_text = ' '.join(comments[comments['polarity'] > 0]['Comment']) if wordcloud_type == "Positive" else ' '.join(comments[comments['polarity'] < 0]['Comment'])

        wordcloud = WordCloud(stopwords=set(STOPWORDS), background_color="white").generate(wordcloud_text)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    elif analysis_type == "Emoji Analysis":
        st.subheader("Emoji Frequency Analysis")

        all_emojis_list = [char for comment in comments['Comment'].dropna() for char in comment if char in emoji.EMOJI_DATA]
        emoji_counts = Counter(all_emojis_list).most_common(10)

        if emoji_counts:
            emojis, freqs = zip(*emoji_counts)
            fig = px.bar(x=emojis, y=freqs, title="Top 10 Emojis Used", labels={'x': 'Emoji', 'y': 'Frequency'}, color_discrete_sequence=['#58508D'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No emojis found in comments.")

# Function for Videos Performance Analysis
def analyze_videos(videos_file):
    df = pd.read_csv(videos_file)
    df.columns = df.columns.str.strip().str.lower()

    st.subheader("🎥 YouTube Videos Performance Analysis")

    fig_size = (5, 2)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Views vs. Likes")
        fig, ax = plt.subplots(figsize=fig_size)
        sns.scatterplot(x=df["views"], y=df["likes"], ax=ax)
        ax.set_title("Views vs. Likes")
        st.pyplot(fig)

    with col2:
        st.write("### Views vs. Comments")
        fig, ax = plt.subplots(figsize=fig_size)
        sns.scatterplot(x=df["views"], y=df["comments"], ax=ax)
        ax.set_title("Views vs. Comments")
        st.pyplot(fig)

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(df[["views", "likes", "comments"]].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    df["engagement_rate"] = (df["likes"] + df["comments"]) / df["views"] * 100

    st.write("### Engagement Rate Distribution")
    fig, ax = plt.subplots(figsize=fig_size)
    sns.histplot(df["engagement_rate"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    top_videos = df.sort_values("engagement_rate", ascending=False).head(10)
    st.write("### Top 10 Engaging Videos")
    st.dataframe(top_videos[["title", "views", "likes", "comments", "engagement_rate"]])
# Function to predict Likes & Comments for a new video
import numpy as np
likes_model = joblib.load("likes_prediction_model.pkl")
comments_model = joblib.load("comments_prediction_model.pkl")

# Function to predict Likes & Comments for a new video
def predict_engagement(views, title):
    if not likes_model or not comments_model:
        st.error("⚠️ Prediction models not found.")
        return 0, 0

    title_length = len(title)
    sentiment_score = TextBlob(title).sentiment.polarity

    # Convert to DataFrame with correct column names as used in training
    features = pd.DataFrame([[views, title_length, sentiment_score]],
                            columns=['Views', 'Title_Length', 'Sentiment_Score'])  # Updated names

    predicted_likes = round(likes_model.predict(features)[0])
    predicted_comments = round(comments_model.predict(features)[0])

    return predicted_likes, predicted_comments



