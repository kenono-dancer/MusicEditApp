import streamlit as st
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import tempfile
import os
import io
import uuid
import time

# Page Config
st.set_page_config(
    page_title="Music Tempo Editor",
    layout="wide"
)

APP_VERSION = "2.0.0"

st.title(f"Music Tempo Editor (v{APP_VERSION})")

# --- Emergency Reset ---
with st.sidebar:
    if st.button("⚠️ Hard Reset App", type="primary", help="Click this if the app crashes or becomes unresponsive. It will clear all data."):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Session State Management ---
if 'tracks' not in st.session_state:
    st.session_state.tracks = []

if 'master_tempo' not in st.session_state:
    st.session_state.master_tempo = 1.0

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# --- Helper Functions ---

def load_audio_segment(file_bytes, file_name):
    """
    Load bytes into Pydub AudioSegment.
    """
    try:
        suffix = os.path.splitext(file_name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        
        # Pydub auto-detects format based on ffmpeg compatibility
        # We explicitly rely on extension or ffmpeg detection
        audio = AudioSegment.from_file(tmp_path)
        
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
        return audio
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None

def add_track_and_reset(uploaded_file):
    try:
        audio = load_audio_segment(uploaded_file.getvalue(), uploaded_file.name)
        if audio:
            track = {
                "id": str(uuid.uuid4()),
                "name": uploaded_file.name,
                "audio": audio, # Store AudioSegment object
                "original_duration_sec": len(audio) / 1000.0,
                # Editing Parameters
                "volume": 0.0, # dB
                "fade_in": 0, # ms
                "fade_out": 0, # ms
                "trim_start": 0.0, # sec
                "trim_end": len(audio) / 1000.0, # sec
            }
            st.session_state.tracks.append(track)
            st.success(f"Added track: {track['name']}")
            
            # Increment key to reset uploader on next rerun
            st.session_state.uploader_key += 1
            time.sleep(0.5) # Short pause to show success message
            st.rerun()
    except Exception as e:
        st.error(f"Failed to add track: {e}")

def duplicate_track(track_id):
    for track in st.session_state.tracks:
        if track["id"] == track_id:
            new_track = track.copy()
            new_track["id"] = str(uuid.uuid4())
            new_track["name"] = f"{track['name']} (Copy)"
            # AudioSegment is immutable-ish but safe to share reference
            st.session_state.tracks.append(new_track)
            st.rerun()
            break

def delete_track(track_id):
    st.session_state.tracks = [t for t in st.session_state.tracks if t["id"] != track_id]
    st.rerun()

def process_track_for_mix(track):
    """
    Apply individual effects (Trim -> Fade -> Volume) to a track.
    Returns: Processed AudioSegment
    """
    try:
        audio = track["audio"]
        
        # 1. Trimming (Pydub uses ms)
        start_ms = int(track["trim_start"] * 1000)
        end_ms = int(track["trim_end"] * 1000)
        
        # Safety check for duration
        if start_ms >= len(audio):
            return None
        if end_ms > len(audio):
            end_ms = len(audio)
        if start_ms >= end_ms:
            return None 
        
        segment = audio[start_ms:end_ms]
        
        # 2. Fade
        if track["fade_in"] > 0:
            # Pydub fade requires fade duration <= segment duration
            fade_in_len = min(track["fade_in"], len(segment))
            segment = segment.fade_in(fade_in_len)
            
        if track["fade_out"] > 0:
            fade_out_len = min(track["fade_out"], len(segment))
            segment = segment.fade_out(fade_out_len)
            
        # 3. Volume
        segment = segment + track["volume"]
        
        return segment
    except Exception as e:
        print(f"Error processing track {track['name']}: {e}")
        return None

def generate_mix():
    """
    Mix all tracks and apply master tempo.
    """
    if not st.session_state.tracks:
        return None, None

    # Mix tracks using Pydub
    mixed_audio = None
    
    for track in st.session_state.tracks:
        segment = process_track_for_mix(track)
        if segment is None:
            continue
            
        if mixed_audio is None:
            mixed_audio = segment
        else:
            mixed_audio = mixed_audio.overlay(segment)
            
    if mixed_audio is None:
        return None, None

    # Export to bytes for Librosa loading
    with io.BytesIO() as buf:
        mixed_audio.export(buf, format="wav")
        buf.seek(0)
        
        # Load into Librosa
        y, sr = librosa.load(buf, sr=44100)

    # Master Tempo processing
    rate = st.session_state.master_tempo
    if rate != 1.0:
        y_final = librosa.effects.time_stretch(y, rate=rate)
    else:
        y_final = y
        
    return y_final, sr

# --- Sidebar / Header Controls ---
with st.container():
    col_up, col_tempo = st.columns([1, 1])
    
    with col_up:
        # Dynamic key to reset uploader after use
        current_key = f"uploader_{st.session_state.uploader_key}"
        uploaded_file = st.file_uploader("Add Track (wav, mp3, m4a)", type=['wav', 'mp3', 'm4a'], key=current_key)
        
        if uploaded_file:
            add_track_and_reset(uploaded_file)
    
    with col_tempo:
        st.subheader("Master Tempo")
        st.session_state.master_tempo = st.slider(
            "Playback Rate (0.500x - 2.000x)",
            min_value=0.500,
            max_value=2.000,
            value=st.session_state.master_tempo,
            step=0.001,
            format="%.3f"
        )

st.markdown("---")

# --- Track List UI ---
st.subheader("Tracks")

if not st.session_state.tracks:
    st.info("No tracks added yet. Upload an audio file to start.")

for i, track in enumerate(st.session_state.tracks):
    # Create a visual card for the track
    with st.container():
        st.markdown(f"#### Track {i+1}")
        
        # Row 1: Name and Basic Actions
        col_name, col_copy, col_del = st.columns([3, 0.5, 0.5])
        with col_name:
            track["name"] = st.text_input("Title", value=track["name"], key=f"name_{track['id']}")
        with col_copy:
            if st.button("©", key=f"dup_{track['id']}", help="Duplicate"):
                duplicate_track(track["id"])
        with col_del:
            if st.button("🗑", key=f"del_{track['id']}", help="Delete"):
                delete_track(track["id"])

        # Row 2: Volume and Fades
        r2_c1, r2_c2, r2_c3 = st.columns(3)
        with r2_c1:
            track["volume"] = st.slider("Volume (dB)", -20.0, 10.0, float(track["volume"]), step=0.5, key=f"vol_{track['id']}")
        with r2_c2:
            track["fade_in"] = st.number_input("Fade In (ms)", 0, 5000, int(track["fade_in"]), step=100, key=f"fi_{track['id']}")
        with r2_c3:
            track["fade_out"] = st.number_input("Fade Out (ms)", 0, 5000, int(track["fade_out"]), step=100, key=f"fo_{track['id']}")

        # Row 3: Trimming
        # Slider range must be within [0, original_duration]
        # value is a tuple (start, end)
        max_dur = track["original_duration_sec"]
        
        # Guard against zero-length or edge cases
        if max_dur > 0:
            track["trim_start"], track["trim_end"] = st.slider(
                "Trim Range (sec)",
                min_value=0.0,
                max_value=max_dur,
                value=(float(track["trim_start"]), float(track["trim_end"])),
                step=0.01,
                key=f"trim_{track['id']}"
            )
        
        st.markdown("---")

# --- Master Mix Preview ---
st.subheader("Master Mix Preview")

if st.button("Generate Mix & Preview", type="primary"):
    if not st.session_state.tracks:
        st.warning("Add at least one track to generate a mix.")
    else:
        with st.spinner("Mixing and Processing..."):
            try:
                y_final, sr = generate_mix()
                
                if y_final is not None:
                    # Convert to WAV for playback
                    buffer = io.BytesIO()
                    sf.write(buffer, y_final, sr, format='WAV', subtype='PCM_16')
                    buffer.seek(0)
                    
                    st.success("Mix Generated Successfully!")
                    st.audio(buffer, format='audio/wav')
                    
                    duration = librosa.get_duration(y=y_final, sr=sr)
                    st.info(f"Total Duration: {duration:.2f} s | Master Tempo: {st.session_state.master_tempo}x")
            
            except Exception as e:
                st.error(f"Error generating mix: {e}")

# Footer
reload_url = f"/?v={int(time.time())}"
st.markdown(
    f"""
    <div style="position: fixed; left: 0; bottom: 0; width: 100%; background-color: #f8f9fa; color: #333; text-align: center; padding: 10px; font-size: 14px; border-top: 1px solid #e9ecef; z-index: 9999;">
        Music Edit App v{APP_VERSION} | 
        <a href="{reload_url}" target="_self" title="Click to update/reload app">Update / Reload</a>
    </div>
    """,
    unsafe_allow_html=True
)
