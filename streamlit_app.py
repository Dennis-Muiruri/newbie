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

# Styling for the Streamlit application
st.set_page_config(page_title="English Accent Classifier", layout="centered")

# Set the GITHUB_RAW_IMAGE_URL to your specific raw GitHub image URL
GITHUB_RAW_IMAGE_URL = "https://raw.githubusercontent.com/Dennis-Muiruri/newbie/main/static/image.png"

st.markdown(f"""
    <style>
    /* Target the main Streamlit app container directly */
    .stApp {{
        background-image: url('{GITHUB_RAW_IMAGE_URL}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        /* Ensure the background covers the full height of the Streamlit app */
        min-height: 100vh;
    }}
    /* Main heading style */
    .main-heading {{
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #38bdf8; /* A shade of blue for vibrancy */
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.8), /* Glow effect */
                                0 0 20px rgba(56, 189, 248, 0.6),
                                0 0 30px rgba(56, 189, 248, 0.4);
        margin-bottom: 30px;
    }}
    /* Style for the text input field */
    .stTextInput > div > div > input {{
        background-color: rgba(255, 255, 255, 0.1); /* Slightly transparent white */
        color: white; /* White text for contrast */
        border: 1px solid rgba(255, 255, 255, 0.3); /* Subtle border */
    }}
    /* Style for the Streamlit button */
    .stButton > button {{
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.9), rgba(67, 56, 202, 0.9)); /* Gradient blue/purple */
        color: white;
        border-radius: 6px;
    }}
    /* Styling for the result display box */
    .result-box {{
        background-color: rgba(15, 23, 42, 0.85); /* Dark, slightly transparent background */
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        color: white; /* White text for results */
    }}
    </style>
""", unsafe_allow_html=True)

# Display the main heading
st.markdown('<h1 class="main-heading">ENGLISH ACCENT CLASSIFIER</h1>', unsafe_allow_html=True)

# --- Accent Analysis Functions ---

@st.cache_resource(show_spinner=False)
def load_classifier():
    """
    Loads the SpeechBrain EncoderClassifier model.
    This function is cached to prevent re-downloading the model on every rerun.
    """
    return EncoderClassifier.from_hparams(
        source=SPEECHBRAIN_ACCENT_MODEL_SOURCE,
        savedir=SPEECHBRAIN_MODEL_SAVEDIR,
        run_opts={"check_for_updates": False} # Prevents frequent checks for model updates
    )

def download_audio(video_url, output_dir):
    """
    Downloads audio from a given video URL using yt-dlp.
    The audio is converted to a WAV file with specific settings (16kHz, mono).
    """
    audio_path = os.path.join(output_dir, "audio.wav")
    audio_path = str(pathlib.Path(audio_path).resolve()) # Ensure absolute path

    command = [
        "yt-dlp",
        "-x", # Extract audio
        "--audio-format", "wav", # Specify WAV format
        "--postprocessor-args", "-ar 16000 -ac 1", # Resample to 16kHz, mono
        "-o", audio_path, # Output path
        video_url, # Input video URL
    ]

    try:
        # Run yt-dlp command to download and process audio
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

        # Classify the accent
        _, score, _, text_lab = classifier.classify_batch(waveform)

        if text_lab:
            label = text_lab[0]
            confidence = float(torch.max(score).exp().item()) * 100 # Convert logit to probability
            classification = ACCENT_MAPPING.get(label, f"Unknown ({label})") # Map label to readable name

            result["classification"] = classification
            result["confidence_score"] = round(confidence, 2)
            result["summary"] = (
                f"The speaker's accent was classified as **{classification}** "
                f"with a confidence score of **{result['confidence_score']}%**."
            )
    except Exception as e:
        result["summary"] = f"Error during analysis: {e}"

    return result

# --- Streamlit UI ---

# Input field for video URL
st.markdown("## Enter a public video URL (e.g., YouTube, Loom, Vimeo)")
video_url = st.text_input("Example: https://www.youtube.com/watch?v=abc123 or https://www.loom.com/share/...", key="video_url_input")

# Analyze button
if video_url and st.button("Analyze Accent"):
    with st.spinner("Downloading and analyzing audio..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = download_audio(video_url, tmpdir)
            if audio_path:
                classifier = load_classifier()
                result = analyze_accent(audio_path, classifier)

                # Display results in a styled box
                st.markdown(f"""
                <div class="result-box">
                    <h2>Accent Analysis Result</h2>
                    <p><strong>Classification:</strong> {result["classification"]}</p>
                    <p><strong>Confidence Score:</strong> {result["confidence_score"]}%</p>
                    <p>{result["summary"]}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Failed to download or process audio. Please check the URL and try again.")
