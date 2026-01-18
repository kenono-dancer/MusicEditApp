import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import tempfile
import os
import io

import time

# Page Config
st.set_page_config(
    page_title="Music Tempo Editor",
    layout="wide"
)

APP_VERSION = "1.2.0"

st.title(f"Music Tempo Editor (v{APP_VERSION})")

@st.cache_data
def load_audio_from_bytes(file_content, file_name):
    """
    Load audio from bytes with caching.
    Uses a temp file for compatibility but returns the numpy array directly.
    """
    try:
        suffix = os.path.splitext(file_name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        # Load audio (always 44.1kHz)
        y, sr = librosa.load(tmp_path, sr=44100)
        
        # Cleanup temp file immediately after loading
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
        return y, sr
    except Exception as e:
        # If cleanup fails or load fails, try to return proper error
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

# 1. File Uploader
uploaded_file = st.file_uploader("Upload Audio (wav, mp3, m4a)", type=['wav', 'mp3', 'm4a'])

if uploaded_file is not None:
    try:
        # 2. Logic: Audio Loading (Cached)
        # We pass file content and name (for extension) to the cached loader
        # This prevents re-loading librosa on every slider change
        with st.spinner("Loading audio..."):
            y, sr = load_audio_from_bytes(uploaded_file.getvalue(), uploaded_file.name)
            
        duration = librosa.get_duration(y=y, sr=sr)
        st.success(f"Loaded: {uploaded_file.name}")
        
        col1, col2 = st.columns(2)
        col1.metric("Sample Rate", f"{sr} Hz")
        col2.metric("Duration", f"{duration:.2f} s")

        # UI: Original Audio Preview
        # We need to write a temp file just for st.audio if we want to play the original?
        # Or we can just play uploaded_file directly!
        st.subheader("Original Audio")
        st.audio(uploaded_file)

        # 3. UI: Tempo Slider
        st.subheader("Tempo Adjustment")
        rate = st.slider(
            "Playback Rate (0.500x - 2.000x)",
            min_value=0.500,
            max_value=2.000,
            value=1.000,
            step=0.001,
            format="%.3f"
        )

        # 4. Logic: Audio Processing & Preview (Auto-run)
        # No button needed. Streamlit reruns on slider change.
        
        # Optimize: Only process if rate is effectively changed or needed
        msg_text = "Processing..."
        if rate == 1.0:
            msg_text = "Preparing preview..."
        
        with st.spinner(msg_text):
            if rate != 1.0:
                y_stretched = librosa.effects.time_stretch(y, rate=rate)
            else:
                y_stretched = y

            # 5. Logic: Convert to 16bit PCM WAV
            buffer = io.BytesIO()
            sf.write(buffer, y_stretched, sr, format='WAV', subtype='PCM_16')
            buffer.seek(0)
            
            # 6. UI: Audio Preview
            st.subheader("Processed Audio")
            st.audio(buffer, format='audio/wav')
            
            # Info about processed audio
            new_duration = librosa.get_duration(y=y_stretched, sr=sr)
            st.info(f"New Duration: {new_duration:.2f} s")

    except Exception as e:
        st.error(f"Error processing audio: {e}")

# Footer Implementation
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa;
        color: #333;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #e9ecef;
        z-index: 9999;
    }
    .footer a {
        color: #0068c9;
        text-decoration: none;
        font-weight: bold;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Cache-busting URL
reload_url = f"/?v={int(time.time())}"

st.markdown(
    f"""
    <div class="footer">
        Music Edit App v{APP_VERSION} | 
        <a href="{reload_url}" target="_self" title="Click to update/reload app">
            Update / Reload
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
