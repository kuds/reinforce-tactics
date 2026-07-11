# Dockerfile for Reinforce Tactics Tournament Runner
# This is a lightweight container for running bot tournaments (no GPU required)

FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Pre-install the heavy dependencies in their own layers so source-only
# changes don't invalidate the download cache. CPU-only torch keeps the
# image lightweight (tournaments need no GPU).
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Remaining base dependencies of the package (without GUI packages)
RUN pip3 install --no-cache-dir \
    gymnasium \
    pettingzoo \
    stable-baselines3 \
    sb3-contrib \
    numpy \
    pandas \
    tensorboard \
    PyYAML

# LLM provider SDKs and Google Cloud Storage for GCS upload support
RUN pip3 install --no-cache-dir \
    openai \
    anthropic \
    google-genai \
    google-cloud-storage

# Copy the game code
COPY . .

# Install the game package (dependencies above already satisfy its
# requirements; the extras make the LLM/GCS support explicit)
RUN pip3 install --no-cache-dir -e ".[llm,cloud]"

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
