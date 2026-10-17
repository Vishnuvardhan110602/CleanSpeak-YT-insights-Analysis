CleanSpeak YT Insights Analysis

CleanSpeak YT Insights Analysis is an AI-powered YouTube comment analysis tool designed to detect, filter, and analyze toxic comments from video discussions. The system helps content creators, moderators, and researchers maintain healthier online communities by identifying harmful interactions and generating actionable insights.

Features
🔍 Toxic Comment Detection using Natural Language Processing (NLP)
🛡️ Automated Filtering of offensive, abusive, hateful, and spam comments
📊 Comment Sentiment Analysis to understand audience engagement
📈 Insightful Analytics Dashboard with toxicity trends and statistics
🎯 Real-time Classification of user comments
📑 Summary Reports for moderation and content improvement
Use Cases
YouTube channel moderation
Community health monitoring
Audience sentiment tracking
Toxicity trend analysis
Research on online behavior and social interactions
Tech Stack
Python
Machine Learning / NLP
YouTube Data API
Pandas & NumPy
Scikit-learn / TensorFlow
Data Visualization Libraries

Project Goal

The goal of CleanSpeak is to promote safer and more positive digital conversations by automatically identifying toxic content and providing meaningful insights into audience behavior.

Steps to Run this repo in local machine

cd CleanSpeak-YT-insights-analysis
install all the required pip libraries

py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install streamlit pandas seaborn matplotlib textblob wordcloud emoji plotly beautifulsoup4 joblib scikit-learn requests pillow
.\.venv\Scripts\python.exe -m streamlit run app.py
