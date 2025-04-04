import streamlit as st
from transformers import pipeline

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter text below to analyze its sentiment.")

# Text input
user_input = st.text_area("Your text:")

if st.button("Analyze"):
    if user_input:
        result = sentiment_pipeline(user_input)[0]
        label = result["label"]
        score = result["score"]
        st.write(f"**Sentiment:** {label} (Confidence: {score:.2f})")
    else:
        st.warning("Please enter some text.")
