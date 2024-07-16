import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torchaudio
from pydub import AudioSegment

# Load Hugging Face model
model_name = "XTTS-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Function to perform text-to-speech synthesis using Hugging Face model
def synthesize_speech(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    with torch.no_grad():
        output = model.generate(input_ids)
    synthesized_audio = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)(output.cpu())
    return synthesized_audio.numpy()

# Streamlit UI
def main():
    st.title('Hugging Face Model Speech Synthesis')
    st.write('Enter text and click the button to synthesize speech using the Hugging Face model (Coqui/XTTS-v2).')

    # Text input widget
    text_input = st.text_area('Enter text to synthesize speech:', height=200)

    if st.button('Synthesize Speech'):
        # Synthesize speech
        synthesized_audio = synthesize_speech(text_input)

        # Convert to compatible format and display
        audio_bytes = synthesized_audio.tobytes()
        st.audio(audio_bytes, format='audio/wav')

if __name__ == '__main__':
    main()
