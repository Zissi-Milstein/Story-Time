import streamlit as st
import pandas as pd
from transformers import pipeline

# Function to perform sentiment analysis
def perform_sentiment_analysis(text):
    sentiment_classifier = pipeline("sentiment-analysis")
    result = sentiment_classifier(text)
    return result[0]['label']

# Streamlit UI
def main():
    st.title('Sentiment Analysis with File Upload')
    st.write('Upload a text file (.txt) for sentiment analysis.')

    # File upload widget
    uploaded_file = st.file_uploader('Choose a file', type=['txt'])

    if uploaded_file is not None:
        # Read the uploaded file
        text = uploaded_file.read().decode('utf-8')

        # Perform sentiment analysis
        sentiment = perform_sentiment_analysis(text)

        # Display results
        st.write('**Sentiment Analysis Result:**')
        st.write(f'Uploaded file sentiment: {sentiment}')

if __name__ == '__main__':
    main()
