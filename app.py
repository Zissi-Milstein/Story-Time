"""
import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import requests
from TTS.api import TTS
os.environ["COQUI_TOS_AGREED"] = "1"

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
github_url = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device) # Replace with your GitHub URL

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
"""
import streamlit as st
import torch
from TTS.utils.generic_utils import setup_model
from TTS.utils.io import load_config
from TTS.vocoder.utils.generic_utils import setup_generator
from TTS.utils.text.symbols import symbols, phonemes
from TTS.utils.synthesis import synthesis
import os

# Function to load the TTS model
def load_tts_model(model_dir):
    # Load TTS model configuration
    model_config = load_config(os.path.join(model_dir, 'config.json'))
    
    # Setup TTS model
    model = setup_model(model_config)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth.tar')))
    model.eval()
    
    # Setup vocoder (if needed)
    vocoder = setup_generator(model_config)
    vocoder.load_state_dict(torch.load(os.path.join(model_dir, 'vocoder_model.pth.tar')))
    vocoder.eval()
    
    return model, vocoder

# Load the TTS model
model_dir = './XTTS-v2'  # Adjust the path as per your directory structure
tts_model, tts_vocoder = load_tts_model(model_dir)

# Function to perform text-to-speech synthesis
def text_to_speech(text):
    with torch.no_grad():
        waveform, _ = synthesis(text, tts_model, tts_vocoder, symbols, phonemes)
    return waveform

# Streamlit UI
def main():
    st.title('Text-to-Speech Synthesis')
    text_input = st.text_input('Enter text to synthesize:', 'Hello, how are you?')

    if st.button('Synthesize'):
        waveform = text_to_speech(text_input)
        st.audio(waveform.numpy(), format='audio/wav')

if __name__ == '__main__':
    main()
