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

# Set the GITHUB_RAW_IMAGE_URL to your specific raw GitHub image URL
# This URL should point directly to your image.png file in your GitHub repository's static folder
GITHUB_RAW_IMAGE_URL = "https://raw.githubusercontent.com/Dennis-Muiruri/newbie/main/static/image.png"

# CSS styling
st.markdown(f"""
<style>
/* Target the main Streamlit app container directly for background */
.stApp {{
    background-image: url('{GITHUB_RAW_IMAGE_URL}');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    margin: 0;
    overflow: hidden;
    min-height: 100vh; /* Ensure the background covers the full height of the Streamlit app */
}}
/* Main heading style */
.main-heading {{
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #38bdf8;
    text-shadow: 0 0 10px rgba(56, 189, 248, 0.8),
                 0 0 20px rgba(56, 189, 248, 0.6);
    margin-bottom: 10px;
}}
.result-box {{
    background-color: rgba(15, 23, 42, 0.85);
    padding: 18px;
    border-radius: 12px;
    color: white;
    font-size: 18px;
    text-align: center;
    margin-top: 20px;
}}
.stTextInput input {{
    background-color: rgba(255, 255, 255, 0.1);
    color: white;
}}
.stButton > button {{
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.9), rgba(67, 56, 202, 0.9));
    color: white;
    border-radius: 8px;
    font-size: 16px;
    padding: 10px 16px;
}}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-heading">ENGLISH ACCENT CLASSIFIER</h1>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_classifier():
    """
    Loads the SpeechBrain EncoderClassifier model.
    This function is cached to prevent re-downloading the model on every rerun.
    """
    return EncoderClassifier.from_hparams(
        source=SPEECHBRAIN_ACCENT_MODEL_SOURCE,
        savedir=SPEECHBRAIN_MODEL_SAVEDIR,
        run_opts={"check_for_updates": False}
    )

def download_audio(video_url, output_dir):
    """
    Downloads audio from a given video URL using yt-dlp.
    The audio is converted to a WAV file with specific settings (16kHz, mono).
    """
    audio_path = os.path.join(output_dir, "audio.wav")
    audio_path = str(pathlib.Path(audio_path).resolve())

    command = [
        "yt-dlp",
        "-x", # Extract audio
        "--audio-format", "wav", # Specify WAV format
        "--postprocessor-args", "-ar 16000 -ac 1", # Resample to 16kHz, mono
        "-o", audio_path, # Output path
        video_url, # Input video URL
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
        return audio_path
    except Exception as e:
        st.error(f"Audio extraction error: {e}")
        return None

def analyze_accent(audio_path, classifier):
    """
    Analyzes the accent from an audio file using the pre-loaded SpeechBrain classifier.
    """
    result = {
        "classification": "Undetermined",
        "confidence_score": 0.0,
        "summary": "Unable to determine accent."
    }

    try:
        # Load the audio waveform
        waveform, sr = torchaudio.load(audio_path)

        # If stereo, convert to mono by taking the mean
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if not already
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        probs, scores, _, labels = classifier.classify_batch(waveform)
        if labels:
            label = labels[0]
            # Original confidence calculation from your previous working code
            confidence = torch.softmax(scores[0], dim=0).max().item() * 100 # Normalize to 0–100%
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
