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

APP_VERSION = "1.1.0"

st.title(f"Music Tempo Editor (v{APP_VERSION})")

def save_uploaded_file(uploaded_file):
    """
    Save uploaded file to a temporary file to ensure library compatibility
    (especially for iOS/Streamlit quirks with direct file objects).
    """
    try:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# 1. File Uploader
uploaded_file = st.file_uploader("Upload Audio (wav, mp3, m4a)", type=['wav', 'mp3', 'm4a'])

if uploaded_file is not None:
    # Save to temp file strictly for loading stability
    temp_path = save_uploaded_file(uploaded_file)
    
    if temp_path:
        try:
            # 2. Logic: Audio Loading (sr=44100)
            with st.spinner("Loading audio..."):
                # librosa.load returns (y, sr)
                # y is a numpy array (float32)
                y, sr = librosa.load(temp_path, sr=44100)
                
            duration = librosa.get_duration(y=y, sr=sr)
            st.success(f"Loaded: {uploaded_file.name}")
            col1, col2 = st.columns(2)
            col1.metric("Sample Rate", f"{sr} Hz")
            col2.metric("Duration", f"{duration:.2f} s")

            # UI: Original Audio Preview
            st.subheader("Original Audio")
            st.audio(temp_path)

            # 3. UI: Tempo Slider
            # Range: 0.500 to 2.000, Step: 0.001
            st.subheader("Tempo Adjustment")
            rate = st.slider(
                "Playback Rate (0.500x - 2.000x)",
                min_value=0.500,
                max_value=2.000,
                value=1.000,
                step=0.001,
                format="%.3f"
            )

            # 4. Logic: Audio Processing & Preview
            if st.button("Apply Tempo Change"):
                with st.spinner("Processing..."):
                    if rate != 1.0:
                        # Time stretch
                        # rate > 1.0 speeds up, rate < 1.0 slows down
                        y_stretched = librosa.effects.time_stretch(y, rate=rate)
                    else:
                        y_stretched = y

                    # 5. Logic: Convert to 16bit PCM WAV
                    # librosa output is float32, usually -1 to 1.
                    # We can use soundfile to write to a memory buffer.
                    
                    # Create in-memory buffer
                    buffer = io.BytesIO()
                    # subtype='PCM_16' ensures 16-bit quality
                    sf.write(buffer, y_stretched, sr, format='WAV', subtype='PCM_16')
                    buffer.seek(0)
                    
                    st.success("Processing Complete!")
                    
                    # 6. UI: Audio Preview
                    st.subheader("Processed Audio")
                    st.audio(buffer, format='audio/wav')
                    
                    # Optional: Info about processed audio
                    new_duration = librosa.get_duration(y=y_stretched, sr=sr)
                    st.info(f"New Duration: {new_duration:.2f} s")

        except Exception as e:
            st.error(f"Error processing audio: {e}")
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

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
