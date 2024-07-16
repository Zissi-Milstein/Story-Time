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
from TTS.api import TTS
import os
import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import requests
from TTS.api import TTS

# Set environment variable
os.environ["COQUI_TOS_AGREED"] = "1"

# Initialize TTS
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model_tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    st.success("Coqui TTS model loaded successfully!")
except Exception as e:
    st.error(f"Error loading Coqui TTS model: {e}")

# Streamlit UI
def main():
    st.title('Voice Clone with Coqui TTS')
    st.markdown("""
        by [Tony Assi](https://www.tonyassi.com/)
        
        Please ❤️ this Space. I build custom AI apps for companies. 
        [Email me](mailto: tony.assi.media@gmail.com) for business inquiries.
    """)

    # Text input
    text_input = st.text_area('Enter text:', height=100)

    # File upload for audio
    uploaded_file = st.file_uploader('Upload voice reference audio file:', type=['wav', 'mp3'])

    if st.button('Clone Voice'):
        if text_input and uploaded_file:
            # Save the uploaded audio file
            audio_path = f"./uploaded_audio.{uploaded_file.name.split('.')[-1]}"
            with open(audio_path, 'wb') as f:
                f.write(uploaded_file.read())

            # Perform voice cloning
            try:
                st.text('Synthesizing...')
                model_tts.tts_to_file(text=text_input, speaker_wav=audio_path, language="en", file_path="./output.wav")
                synthesized_audio = open("./output.wav", 'rb').read()
                st.audio(synthesized_audio, format='audio/wav')
            except Exception as e:
                st.error(f"Error synthesizing voice: {e}")
        else:
            st.warning('Please provide both text and a voice reference audio file.')

if __name__ == '__main__':
    main()
