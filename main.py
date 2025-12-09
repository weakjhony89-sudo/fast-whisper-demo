import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import subprocess
import json
import os
from datetime import datetime

# -----------------------------
# Load Faster Whisper Model
# -----------------------------
@st.cache_resource
def load_model():
    # small, fast model
    return WhisperModel("guillaumekln/faster-whisper-small", device="cpu")

model = load_model()

# -----------------------------
# Extract Audio from Video
# -----------------------------
def extract_audio(video_path, audio_path):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                   # no video
        "-acodec", "pcm_s16le",  # wav
        "-ar", "16000",          # sample rate
        audio_path,
        "-y"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎥 Video → Audio → Text (Faster Whisper)")
st.write("Upload a video, extract audio, and transcribe with Faster Whisper.")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file:
    st.video(uploaded_file)

    # Save uploaded video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        tmp_video.write(uploaded_file.read())
        video_path = tmp_video.name

    st.info("Extracting audio...")
    with st.spinner("Processing video…"):
        audio_path = video_path + ".wav"
        extract_audio(video_path, audio_path)

    st.success("Audio extracted!")
    st.audio(audio_path)

    # --- Transcription ---
    st.info("Running Faster Whisper…")
    start_time = datetime.now()
    with st.spinner("Transcribing audio…"):
        segments, info = model.transcribe(audio_path)

    # --------------- TEXT FORMAT --------------------
    st.subheader("📝 Text Transcription")
    text_output = " ".join([seg.text for seg in segments])
    end_time = datetime.now()
    execution_time = end_time - start_time

    
    st.write(text_output)
    st.write(f"**Transcription Time:** {execution_time.total_seconds():.2f} seconds")

    # --------------- Download Buttons ---------------
    st.download_button(
        "⬇️ Download Text",
        data=text_output,
        file_name="transcription.txt",
        mime="text/plain"
    )
