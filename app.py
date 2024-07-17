
import streamlit as st
from TTS.api import TTS
from TTS.utils.manage import ModelManager
from TTS.utils.generic_utils import get_user_data_dir
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig
import os
import torch
# import spaces
# Set environment variable if needed (e.g., for Coqui TTS)
os.environ["COQUI_TOS_AGREED"] = "1"

# Initialize TTS model
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model_names = TTS().list_models()
    # print(model_names.__dict__)
    # print(model_names.__dir__())
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2" 
    ModelManager().download_model(model_name)
    st.success("Coqui TTS model loaded successfully!")
except Exception as e:
    st.error(f"Error downloading Coqui TTS model: {e}")

# model_dir = "https://github.com/Zissi-Milstein/StoryTime/tree/main/XTTS-v2" 
# @spaces.GPU(enable_queue=True)
try:
    
    tts = TTS(model_name, gpu=False)
    tts.to("device")
    # tts = TTS("coqui/XTTS-v2").to(device)
    # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    # tts = TTS(model_path="XTTS-v2/model.pth", config_path="XTTS-v2/config.json", progress_bar=False, gpu=False)
    st.success("Coqui TTS model loaded successfully!")
except Exception as e:
    st.error(f"Error loading Coqui TTS model: {e}")
    # tts = None  # Set tts to None if initialization fails

# Function to synthesize speech
def clone_voice(text_input, uploaded_file):
    # if tts is None:
        # st.error("TTS model not loaded. Please check the model initialization.")
        # return None

    # Save the uploaded audio file
    audio_path = f"./uploaded_audio.{uploaded_file.name.split('.')[-1]}"
    with open(audio_path, 'wb') as f:
        f.write(uploaded_file.read())
        
    st.text('Synthesizing...')
    # synthesized_audio = tts(text_input)[0]['audio']
    # st.audio(synthesized_audio, format='audio/wav')
    try:
        # tts.tts_to_file(text=text_input, language="en", file_path="./output.wav")
        # wav = tts.tts("This is a test! This is also a test!!", speaker=tts.uploaded_file, language=tts.languages[0])
        tts.tts(text=text_input, speaker_wav=uploaded_file)
        # tts.tts_to_file(text=text_input, speaker_wav=uploaded_file, language="en", file_path="./output.wav")
        # return "./output.wav"
    except Exception as e:
        st.error(f"Error synthesizing voice: {e}")


# Streamlit UI
def main():
    st.title('Text-to-Speech with Coqui TTS (xtts_v2)')
    st.markdown("""
        by Sarah Milstein
        
        This app uses Coqui TTS to synthesize speech from text input.
    """)

    # Text input area
    # Text input
    text_input = st.text_area('Enter text:', height=100)

    # File upload for audio
    uploaded_file = st.file_uploader('Upload voice reference audio file:', type=['wav', 'mp3'])

    if st.button('Clone Voice') and text_input:
        if uploaded_file:
            clone_voice(text_input, uploaded_file)
        else:
            st.error('Please upload a voice reference audio file.')

if __name__ == '__main__':
    main()
