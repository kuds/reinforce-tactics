FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy game code
COPY . .

# Install game with cloud extras (google-cloud-storage for artifact sync)
RUN pip3 install --no-cache-dir -e ".[cloud]"

# The entrypoint wraps the training command with periodic + final upload of
# outputs (models/, checkpoints/, tensorboard/, logs/) to Google Cloud Storage
# when GCS_OUTPUT_URI or Vertex's AIP_MODEL_DIR is set. Arguments after the
# entrypoint are the training command to run; CMD provides a sensible default.
# To bypass the wrapper for debugging, override it: docker run --entrypoint bash ...
ENTRYPOINT ["python3", "scripts/cloud/vertex_train.py"]
CMD ["python3", "main.py", "--mode", "train"]
