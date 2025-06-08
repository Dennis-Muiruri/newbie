import os
import tempfile
import subprocess
import pathlib
import torch
import torchaudio
import streamlit as st
from speechbrain.inference.classifiers import EncoderClassifier

# --------------------------
# Config
# --------------------------
SPEECHBRAIN_ACCENT_MODEL_SOURCE = "Jzuluaga/accent-id-commonaccent_ecapa"
SPEECHBRAIN_MODEL_SAVEDIR = os.path.join(os.path.expanduser("~"), "speechbrain_models")

ACCENT_MAPPING = {
    "african": "African English",
    "australia": "Australian English",
    "bermuda": "Bermudan English",
    "canada": "Canadian English",
    "england": "British English (England)",
    "hongkong": "Hong Kong English",
    "indian": "Indian English",
    "ireland": "Irish English",
    "malaysia": "Malaysian English",
    "newzealand": "New Zealand English",
    "philippines": "Philippine English",
    "scotland": "Scottish English",
    "singapore": "Singaporean English",
    "southatlandtic": "South Atlantic English",
    "us": "American English (US)",
    "wales": "Welsh English",
}

# --------------------------
# Functions
# --------------------------
@st.cache_resource
def load_classifier():
    return EncoderClassifier.from_hparams(
        source=SPEECHBRAIN_ACCENT_MODEL_SOURCE,
        savedir=SPEECHBRAIN_MODEL_SAVEDIR,
        run_opts={"check_for_updates": False}
    )

def download_audio(video_url, output_dir):
    audio_path = os.path.join(output_dir, "audio.wav")
    audio_path = str(pathlib.Path(audio_path).resolve())

    command = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--postprocessor-args", "-ar 16000 -ac 1",
        "-o", audio_path,
        video_url,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
        return audio_path
    except Exception as e:
        st.error(f"Audio extraction error: {e}")
        return None

def analyze_accent(audio_path, classifier):
    try:
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        _, score, _, text_lab = classifier.classify_batch(waveform)

        if text_lab:
            label = text_lab[0]
            confidence = float(torch.max(score).exp().item()) * 100
            classification = ACCENT_MAPPING.get(label, f"Unknown ({label})")
            summary = f"The speaker's accent was classified as **{classification}** with a confidence score of **{confidence:.2f}%**."

            return classification, confidence, summary
    except Exception as e:
        return "Undetermined", 0.0, f"Error during analysis: {e}"

    return "Undetermined", 0.0, "Unable to determine accent."

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="English Accent Classifier", layout="centered")
st.title("🗣️ English Accent Classifier")
st.markdown("Analyze a speaker's English accent from a public YouTube video.")

video_url = st.text_input("Enter YouTube Video URL:", placeholder="https://...")

if st.button("Analyze Accent") and video_url:
    with st.spinner("Downloading and analyzing audio..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = download_audio(video_url, tmpdir)
            if audio_path:
                classifier = load_classifier()
                classification, confidence, summary = analyze_accent(audio_path, classifier)
                st.success("Analysis Complete ✅")
                st.markdown(f"**Classification**: {classification}")
                st.markdown(f"**Confidence Score**: {confidence:.2f}%")
                st.markdown(summary)
            else:
                st.error("Failed to download or process audio.")
