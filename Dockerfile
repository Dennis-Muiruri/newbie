# Use a Python base image with common build tools
FROM python:3.10-slim-buster # Python 3.10 is often more stable for ML dependencies

# Install system dependencies required for sentencepiece, yt-dlp, etc.
# These are common for build environments on Debian/Ubuntu-based images.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install yt-dlp
RUN pip install yt-dlp

# Set the working directory
WORKDIR /app

# Copy your requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir can save space but might slow down rebuilds slightly
RUN pip install -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port Streamlit runs on (default is 8501)
EXPOSE 8501

# Command to run your Streamlit application
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.enableCORS", "false"]