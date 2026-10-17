




import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from analysis import analyze_comments, analyze_videos, predict_engagement

# Set Page Config
st.set_page_config(page_title="YouTube Analysis", layout="wide")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
section = st.sidebar.radio("Go to", ["Upload & Overview", "Comments Analysis", "Videos Performance", "Predict Engagement"])

# Title
st.title("📊 YouTube Insights Analysis")

# Load and Display Image
image_url = "https://www.kindpng.com/picc/m/432-4328812_youtube-analytics-you-tube-logo-hd-png-download.png"
try:
    response = requests.get(image_url, timeout=5)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        st.image(image, use_container_width=False)
except requests.exceptions.RequestException:
    st.warning("⚠️ Failed to load image. Please check your internet connection.")

# Sidebar for Uploads
st.sidebar.subheader("📤 Upload Data")
comments_file = st.sidebar.file_uploader("Upload a CSV file for Comments Analysis", type=["csv"])
videos_file = st.sidebar.file_uploader("Upload a CSV file for Videos Performance Analysis", type=["csv"])

# Process Sections
if section == "Comments Analysis":
    if comments_file:
        analyze_comments(comments_file)
    else:
        st.warning("⚠️ Please upload a CSV file to analyze comments.")

elif section == "Videos Performance":
    if videos_file:
        analyze_videos(videos_file)
    else:
        st.warning("⚠️ Please upload a CSV file to analyze video performance.")

elif section == "Predict Engagement":
    st.subheader("📈 Predict Likes & Comments for a New Video")
    expected_views = st.number_input("Enter Expected Views", min_value=0, step=1000)
    video_title = st.text_input("Enter the Video Title")

    if st.button("Predict"):
        if video_title.strip():
            try:
                predicted_likes, predicted_comments = predict_engagement(expected_views, video_title)
                st.success(f"📌 Predicted Likes: {predicted_likes}")
                st.success(f"💬 Predicted Comments: {predicted_comments}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("⚠️ Please enter a video title!")
