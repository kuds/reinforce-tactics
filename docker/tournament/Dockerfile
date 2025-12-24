# Dockerfile for Reinforce Tactics Tournament Runner
# This is a lightweight container for running bot tournaments (no GPU required)

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies (without GUI/GPU packages)
RUN pip3 install --no-cache-dir \
    gymnasium \
    pettingzoo \
    numpy \
    pandas \
    argparse \
    typing-extensions

# Install LLM provider SDKs
RUN pip3 install --no-cache-dir \
    openai \
    anthropic \
    google-genai

# Install Google Cloud Storage for GCS upload support
RUN pip3 install --no-cache-dir google-cloud-storage

# Copy the game code
COPY . .

# Install the game package
RUN pip3 install -e .

# Create output directories
RUN mkdir -p /app/output/results /app/output/conversations /app/output/replays /app/config

# Copy the tournament runner script
COPY docker/tournament/run_tournament.py /app/run_tournament.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default config location
ENV TOURNAMENT_CONFIG=/app/config/config.json

# Entry point
ENTRYPOINT ["python3", "/app/run_tournament.py"]
CMD ["--config", "/app/config/config.json"]
