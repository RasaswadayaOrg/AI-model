# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

# System deps (libgomp for torch, build tools for torch-geometric extras)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (smaller image than default CUDA build)
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Then the rest
COPY requirements.txt ./
RUN pip install --no-cache-dir \
    "torch-geometric>=2.4.0" \
    fastapi==0.115.4 \
    "uvicorn[standard]==0.32.0" \
    networkx \
    scipy \
    numpy \
    pandas \
    scikit-learn \
    tqdm \
    python-dotenv \
    psycopg2-binary \
    supabase \
    matplotlib \
    seaborn

# Copy application
COPY . .

# Hugging Face Spaces routes traffic to $PORT (default 7860)
EXPOSE 7860

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}"]
