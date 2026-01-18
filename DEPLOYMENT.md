# Deployment Instructions

> [!IMPORTANT]
> **CRITICAL: Python Version Setting**
> This app requires **Python 3.11**.
> Python 3.13 (often the default now) causes the crash you are seeing because the audio library (`numba`/`librosa`) does not support it yet.
>
> **How to fix on Streamlit Cloud:**
> 1. Go to your App Dashboard (share.streamlit.io).
> 2. Click the specific app settings (three dots) -> **Settings**.
> 3. Under **General**, change **Python version** to **3.11**.
> 4. Save/Reboot the app.

Your code is successfully hosted on GitHub at:
**[https://github.com/kenono-dancer/MusicEditApp](https://github.com/kenono-dancer/MusicEditApp)**

## Deploy to Streamlit Cloud (Recommended / Free)
Streamlit Cloud is the easiest way to host this application.

1.  **Go to Streamlit Cloud**:
    - Visit [https://share.streamlit.io/](https://share.streamlit.io/) and signing with your GitHub account.

2.  **Create New App**:
    - Click **"New app"** (usually top right).

3.  **Connect Repository**:
    - **Repository**: Select `kenono-dancer/MusicEditApp`.
    - **Branch**: `main`.
    - **Main file path**: `app.py`.
    - **Advanced Settings**: Select **Python 3.11**.

4.  **Deploy**:
    - Click **"Deploy!"**.
    - Wait 1-2 minutes for the build to finish.

## Requirements Check
The repository already contains the necessary configuration files:
- `requirements.txt`: Installs `streamlit`, `librosa`, etc.
- `packages.txt`: Installs `ffmpeg` (Required for audio processing).

## Troubleshooting
- If deployment fails, check the logs in the Streamlit Cloud dashboard.
- Common issues:
    - `FFmpeg not found`: Ensure `packages.txt` exists (it does).
    - Memory limits: Large audio files might crash the free tier instance.
