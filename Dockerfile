FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for insightface + opencv/mediapipe + ffmpeg + faster-whisper runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    build-essential \
    g++ \
    cmake \
    pkg-config \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps (make sure requirements.txt includes faster-whisper)
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Default shell (compose will override command)
COPY credentials.json /app/credentials.json
COPY token.json /app/token.json
CMD ["bash"]