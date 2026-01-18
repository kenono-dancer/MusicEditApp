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

import shutil
import imageio_ffmpeg
import subprocess

APP_VERSION = "2.0.5"

# --- FFMPEG Configuration ---
# 1. Try system ffmpeg
ffmpeg_path = shutil.which("ffmpeg")
ffprobe_path = shutil.which("ffprobe")

# 2. Key Fallback: Use imageio-ffmpeg if available (Guaranteed Binary)
if not ffmpeg_path:
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"Using imageio-ffmpeg binary: {ffmpeg_exe}")
        
        # CRITICAL: Create an alias named 'ffmpeg' because 'audioread' looks for that specific command
        # We create a local 'bin' folder and copy the binary there.
        work_dir = os.getcwd()
        bin_dir = os.path.join(work_dir, "bin")
        os.makedirs(bin_dir, exist_ok=True)
        
        target_ffmpeg = os.path.join(bin_dir, "ffmpeg")
        
        if not os.path.exists(target_ffmpeg):
            shutil.copy(ffmpeg_exe, target_ffmpeg)
            # Make sure it's executable
            st = os.stat(target_ffmpeg)
            os.chmod(target_ffmpeg, st.st_mode | 0o111)
            
        # Add this bin dir to PATH
        os.environ["PATH"] += os.pathsep + bin_dir
        
        # Update paths for valid checks
        ffmpeg_path = target_ffmpeg
        
    except Exception as e:
        print(f"imageio-ffmpeg setup failed: {e}")

# Set Pydub paths
if ffmpeg_path:
    AudioSegment.converter = ffmpeg_path
    # Usually ffprobe is in the same dir ??
    # Imageio-ffmpeg doesn't strictly provide ffprobe, but Pydub mainly needs converter for transcode.
    # For probe, it might fail if we don't have it.
    
if ffprobe_path:
    AudioSegment.ffprobe = ffprobe_path

# Display Debug Info in Sidebar (Temporary)
with st.sidebar:
    st.caption(f"Backend Info (v2.0.2):")
    st.caption(f"- FFMPEG: {ffmpeg_path or 'Not Found'}")
    st.caption(f"- FFPROBE: {ffprobe_path or 'Not Found'}")

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

# Automation State
if 'automation_is_recording' not in st.session_state:
    st.session_state.automation_is_recording = False
if 'automation_data' not in st.session_state:
    st.session_state.automation_data = [] # List of (timestamp, rate)
if 'automation_start_time' not in st.session_state:
    st.session_state.automation_start_time = None

# --- Helper Functions ---

def load_audio_segment(file_bytes, file_name):
    """
    Load bytes into Pydub AudioSegment.
    Tries Pydub first (ffmpeg), then Librosa (soundfile/audioread) as fallback.
    """
    # 1. Create temp file
    try:
        suffix = os.path.splitext(file_name)[1]
        if not suffix:
            suffix = ".mp3" # default guess
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
    except Exception as e:
        st.error(f"Error creating temp file: {e}")
        return None

    
    # 2. Try Loading
    audio = None
    
    # Attempt 1: Pydub (Direct - depends on system ffmpeg/ffprobe)
    try:
        audio = AudioSegment.from_file(tmp_path)
    except Exception as e_pydub:
        print(f"Pydub load failed: {e_pydub}. Trying Fallback...")
        
        # Attempt 2: Manual FFMPEG Conversion (Bypasses ffprobe & Librosa/audioread)
        # We use our guaranteed 'ffmpeg_path' to transcode to WAV, then load WAV.
        try:
            if not ffmpeg_path:
                raise Exception("No FFMPEG binary found for fallback.")
                
            wav_path = tmp_path + ".wav"
            
            # Run ffmpeg command: ffmpeg -i input -y output.wav
            # -y overwrites, -v error (quiet)
            cmd = [
                ffmpeg_path, 
                "-i", tmp_path,
                "-f", "wav",
                "-y", 
                "-v", "error",
                wav_path
            ]
            
            print(f"Running fallback cmd: {cmd}")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Load the WAV (Pydub handles WAV cleanly without external tools usually)
            audio = AudioSegment.from_file(wav_path, format="wav")
            print("Manual FFMPEG fallback successful.")
            
            # Cleanup WAV
            if os.path.exists(wav_path):
                os.remove(wav_path)
                
        except Exception as e_convert:
            st.error(f"Audio Load Error.\n1. Pydub: {e_pydub}\n2. Manual FFMPEG: {e_convert}")
            # Try Librosa as last ditch (though likely to fail if manual ffmpeg failed)
            # Leaving it out to keep error clear.
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return None

    # Cleanup
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        
    return audio

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

def generate_dynamic_mix():
    """
    Mix tracks and apply Dynamic Tempo based on automation_data.
    """
    if not st.session_state.tracks:
        return None, None

    # 1. Mix tracks (Base Mix)
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

    # Export to bytes for Librosa
    with io.BytesIO() as buf:
        mixed_audio.export(buf, format="wav")
        buf.seek(0)
        y, sr = librosa.load(buf, sr=44100)

    # 2. Apply Tempo
    # If Automation Data exists, use Variable Rate
    if st.session_state.automation_data:
        points = sorted(st.session_state.automation_data, key=lambda x: x[0])
        
        # Make sure we cover the start
        if not points or points[0][0] > 0:
             # Default to current slider value or 1.0 for the start
             initial_rate = 1.0 
             points.insert(0, (0.0, initial_rate))
             
        outputs = []
        total_samples = len(y)
        
        for i in range(len(points)):
            t_start, rate = points[i]
            
            # Determine end time for this segment
            if i + 1 < len(points):
                t_end = points[i+1][0]
            else:
                t_end = len(y) / sr # End of file
                
            start_sample = int(t_start * sr)
            end_sample = int(t_end * sr)
            
            # Safety bounds
            start_sample = max(0, min(start_sample, total_samples))
            end_sample = max(0, min(end_sample, total_samples))
            
            if start_sample >= end_sample:
                continue
                
            chunk = y[start_sample:end_sample]
            
            # Time Stretch
            # Note: Librosa time_stretch is quality but slow for many small chunks
            if rate != 1.0:
                # Speed up or slow down
                chunk_stretched = librosa.effects.time_stretch(chunk, rate=rate)
            else:
                chunk_stretched = chunk
                
            outputs.append(chunk_stretched)
            
        if outputs:
            y_final = np.concatenate(outputs)
        else:
            y_final = y

    # Else use Constant Master Tempo
    else:
        rate = st.session_state.master_tempo
        if rate != 1.0:
            y_final = librosa.effects.time_stretch(y, rate=rate)
        else:
            y_final = y
            
    return y_final, sr

def on_tempo_slider_change():
    """Callback for slider movement."""
    if st.session_state.automation_is_recording:
        # Check if we have a start time, if not (first move), set it?
        # Actually start time should be set when 'Record' is toggled ON? 
        # No, usually regular playback starts THEN we record?
        # Let's assume 'automation_start_time' was set when user clicked 'Start Recording' 
        # OR we use the current time relative to when recording was ENABLED.
        
        if st.session_state.automation_start_time is None:
             st.session_state.automation_start_time = time.time()
             
        elapsed = time.time() - st.session_state.automation_start_time
        new_rate = st.session_state.master_tempo
        
        # Append point
        st.session_state.automation_data.append((elapsed, new_rate))


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
        
        # automation controls
        c1, c2, c3 = st.columns([0.4, 0.3, 0.3])
        with c1:
            # Toggle Recording
            if st.checkbox("🔴 Record Automation", value=st.session_state.automation_is_recording, key="record_toggle"):
                if not st.session_state.automation_is_recording:
                    # Just started recording
                    st.session_state.automation_is_recording = True
                    st.session_state.automation_start_time = time.time()
                    st.toast("Recording Started! Move slider to capture points.")
            else:
                if st.session_state.automation_is_recording:
                    # Just stopped
                    st.session_state.automation_is_recording = False
                    st.toast("Recording Stopped.")
        
        with c2:
            if st.button("Clear Data"):
                st.session_state.automation_data = []
                st.rerun()
                
        with c3:
            st.metric("Points", len(st.session_state.automation_data))

        # Slider with Callback
        st.session_state.master_tempo = st.slider(
            "Playback Rate (0.500x - 2.000x)",
            min_value=0.500,
            max_value=2.000,
            value=st.session_state.master_tempo,
            step=0.001,
            format="%.3f",
            on_change=on_tempo_slider_change
        )
        # Note regarding 'on_change': Streamlit usually triggers this on release.
        # For real-time recording, users must release mouse or use keyboard for updates?
        # Actually slider *does* trigger on drag release. Continuous updates might require a custom component,
        # but this is "standard" Streamlit behavior.

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

        # Waveform & Playback
        try:
            # 1. Playback
            # We need to export to bytes. For performance, do this on demand?
            # Streamlit re-runs script, so it might be okay.
            # To avoid lag, we could cache this? But audio is mutable (trim/fade).
            # Let's just do it.
            
            # Use original audio for preview without effects? Or with effects?
            # User probably wants to hear effects (volume/fade).
            # But effects are applied in 'process_track_for_mix'.
            
            # Let's show the *processed* preview? Or raw?
            # Usually individual track preview is raw or with basic volume.
            # Let's show the raw loaded audio (cut to trim range?) for now, 
            # or maybe just the full file.
            # User said "Play button for each track" -> usually implies listening to that source.
            
            audio_source = track["audio"]
            
            # Create a localized buffer for this track
            with io.BytesIO() as track_buf:
                audio_source.export(track_buf, format="wav")
                st.audio(track_buf.getvalue(), format='audio/wav')
            
            # 2. Waveform Visualization
            # Convert to numpy for display
            # Pydub samples -> numpy
            samples = np.array(audio_source.get_array_of_samples())
            
            # Handle Stereo (interleaved)
            if audio_source.channels == 2:
                # Reshape to (samples, 2)
                samples = samples.reshape((-1, 2))
                # Mix down to mono for display or just take left channel
                samples = samples.mean(axis=1) # Average
                
            # Downsample for performance (aim for ~500-1000 points)
            # If 3min song @ 44.1k = ~8M samples.
            step = max(1, len(samples) // 1000)
            view_data = samples[::step]
            
            # Normalize for better visual
            if len(view_data) > 0:
                view_data = view_data / (np.max(np.abs(view_data)) + 1e-9)
            
            st.area_chart(view_data, height=60, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Preview unavailable: {e}")

        
        # Row 2: Trim Controls (Moved Up)
        max_dur = track["original_duration_sec"]
        if max_dur > 0:
            # Info Display
            t_start, t_end = track["trim_start"], track["trim_end"]
            
            # Slider
            c_info, c_slide = st.columns([1, 2])
            with c_info:
                 st.caption(f"⏱️ Start: {t_start:.2f}s | End: {t_end:.2f}s | Duration: {t_end-t_start:.2f}s")
                 
            track["trim_start"], track["trim_end"] = st.slider(
                "Trim Range (sec)",
                min_value=0.0,
                max_value=max_dur,
                value=(float(track["trim_start"]), float(track["trim_end"])),
                step=0.01,
                key=f"trim_{track['id']}"
            )
            
            # Preview Button for Trimmed Segment
            # Calculate range
            p_start = int(track["trim_start"] * 1000)
            p_end = int(track["trim_end"] * 1000)
            
            if p_end > p_start:
                 # Slice audio (raw, no effects yet, just trim)
                 trim_audio = track["audio"][p_start:p_end]
                 
                 # Export to bytes
                 with io.BytesIO() as trim_buf:
                     trim_audio.export(trim_buf, format="wav")
                     st.audio(trim_buf.getvalue(), format='audio/wav')
            


        # Row 3: Volume and Fades (Moved Down)
        r3_c1, r3_c2, r3_c3 = st.columns(3)
        with r3_c1:
            track["volume"] = st.slider("Volume (dB)", -20.0, 10.0, float(track["volume"]), step=0.5, key=f"vol_{track['id']}")
        with r3_c2:
            track["fade_in"] = st.number_input("Fade In (ms)", 0, 5000, int(track["fade_in"]), step=100, key=f"fi_{track['id']}")
        with r3_c3:
            track["fade_out"] = st.number_input("Fade Out (ms)", 0, 5000, int(track["fade_out"]), step=100, key=f"fo_{track['id']}")

        
        st.markdown("---")

# --- Master Mix Preview ---
st.subheader("Master Mix Preview")

if st.button("Generate Mix & Preview", type="primary"):
    if not st.session_state.tracks:
        st.warning("Add at least one track to generate a mix.")
    else:
        with st.spinner("Mixing and Processing..."):
            try:
                # Use dynamic mix function
                y_final, sr = generate_dynamic_mix()
                
                if y_final is not None:
                    # Convert to WAV for playback
                    buffer = io.BytesIO()
                    sf.write(buffer, y_final, sr, format='WAV', subtype='PCM_16')
                    buffer.seek(0)
                    
                    st.success("Mix Generated Successfully!")
                    st.audio(buffer, format='audio/wav')
                    
                    # Duration & Info
                    duration = librosa.get_duration(y=y_final, sr=sr)
                    info_text = f"Total Duration: {duration:.2f} s"
                    if st.session_state.automation_data:
                        info_text += f" | Automation Points: {len(st.session_state.automation_data)}"
                    else:
                        info_text += f" | Static Tempo: {st.session_state.master_tempo}x"
                    st.info(info_text)

                    # Export Button (High Quality Download)
                    st.download_button(
                        label="⬇️ Download High-Quality WAV",
                        data=buffer,
                        file_name="remix_master.wav",
                        mime="audio/wav"
                    )
            
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
