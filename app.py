import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import requests

# Function to load tokenizer and model from GitHub
def load_model_from_github(github_url):
    try:
        tokenizer = AutoTokenizer.from_pretrained(github_url, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(github_url)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model from GitHub: {e}")
        return None, None

# Define the GitHub raw URL where your model files are hosted
github_url = "https://github.com/Zissi-Milstein/Story-Time/blob/main/XTTS-v2/model.pth"  # Replace with your GitHub URL

# Load tokenizer and model from GitHub
tokenizer, model = load_model_from_github(github_url)

# Function to perform text-to-speech synthesis using Hugging Face model
def synthesize_speech(text):
    if tokenizer is None or model is None:
        st.warning("Model not loaded. Check the error message above.")
        return None

    try:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs)
        synthesized_audio = output.to('cpu').numpy()
        return synthesized_audio
    except Exception as e:
        st.error(f"Error synthesizing speech: {e}")
        return None

# Streamlit UI
def main():
    st.title('GitHub Model Speech Synthesis')
    st.write('Enter text and click the button to synthesize speech using the Hugging Face model loaded from GitHub.')

    # Text input widget
    text_input = st.text_area('Enter text to synthesize speech:', height=200)

    if st.button('Synthesize Speech'):
        # Synthesize speech
        synthesized_audio = synthesize_speech(text_input)

        if synthesized_audio is not None:
            # Convert to compatible format and display
            st.audio(synthesized_audio, format='audio/wav')

if __name__ == '__main__':
    main()
