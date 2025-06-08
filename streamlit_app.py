import os
import tempfile
import subprocess
import pathlib
import torch
import torchaudio
import streamlit as st
from speechbrain.inference.classifiers import EncoderClassifier

# Constants
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

# Page config
st.set_page_config(
    page_title="English Accent Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS styling
st.markdown("""
<style>
body {
    background-image: url('https://raw.githubusercontent.com/Dennis-Muiruri/newbie/refs/heads/main/static/image.png');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    margin: 0;
    overflow: hidden;
}
.main-heading {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #38bdf8;
    text-shadow: 0 0 10px rgba(56, 189, 248, 0.8),
                 0 0 20px rgba(56, 189, 248, 0.6);
    margin-bottom: 10px;
}
.result-box {
    background-color: rgba(15, 23, 42, 0.85);
    padding: 18px;
    border-radius: 12px;
    color: white;
    font-size: 18px;
    text-align: center;
    margin-top: 20px;
}
.stTextInput input {
    background-color: rgba(255, 255, 255, 0.1);
    color: white;
}
.stButton > button {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.9), rgba(67, 56, 202, 0.9));
    color: white;
    border-radius: 8px;
    font-size: 16px;
    padding: 10px 16px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-heading">ENGLISH ACCENT CLASSIFIER</h1>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
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
    result = {
        "classification": "Undetermined",
        "confidence_score": 0.0,
        "summary": "Unable to determine accent."
    }

    try:
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        probs, scores, _, labels = classifier.classify_batch(waveform)
        if labels:
            label = labels[0]
            confidence = torch.softmax(scores[0], dim=0).max().item() * 100  # Normalize to 0–100%
            classification = ACCENT_MAPPING.get(label, f"Unknown ({label})")

            result["classification"] = classification
            result["confidence_score"] = round(confidence, 2)
            result["summary"] = (
                f"The speaker's accent was classified as <b>{classification}</b> "
                f"with a confidence score of <b>{result['confidence_score']}%</b>."
            )
    except Exception as e:
        result["summary"] = f"Error during analysis: {e}"

    return result

# Input UI
video_url = st.text_input("Enter a public YouTube video URL (e.g., https://www.youtube.com/watch?v=xyz123)")

# Action button
if video_url and st.button("Analyze Accent"):
    with st.spinner("Analyzing audio..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = download_audio(video_url, tmpdir)
            if audio_path:
                classifier = load_classifier()
                result = analyze_accent(audio_path, classifier)

                st.markdown(f"""
                <div class="result-box">
                    <h3>Accent Analysis Result</h3>
                    <p><strong>Classification:</strong> {result["classification"]}</p>
                    <p><strong>Confidence Score:</strong> {result["confidence_score"]}%</p>
                    <p>{result["summary"]}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Failed to process audio. Check your video URL.")
