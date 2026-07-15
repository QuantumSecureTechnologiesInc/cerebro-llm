# Cerebro LLM Production Docker Image
# Multi-stage build for minimal image size
# Base: PyTorch CUDA with Python 3.11

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e ".[server]" && \
    pip install --no-cache-dir tiktoken safetensors tqdm numpy

# Copy source code
COPY cerebro/ /app/cerebro/

# Create non-root user
RUN useradd --create-home --shell /bin/bash cerebro && \
    mkdir -p /app/checkpoints /app/data && \
    chown -R cerebro:cerebro /app
USER cerebro

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: start API server
ENV CEREBRO_MODEL=nano
ENTRYPOINT ["python", "-m", "cerebro.cli"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]


# ── Development image ──
FROM base AS dev
USER root
RUN pip install --no-cache-dir "pytest>=7.0" "pytest-cov>=4.0" "pytest-asyncio>=0.23"
USER cerebro


# ── Training image (larger, with all training deps) ──
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel AS training

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git htop nvtop \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e ".[all]"

COPY cerebro/ /app/cerebro/
RUN useradd --create-home --shell /bin/bash cerebro && \
    mkdir -p /app/checkpoints /app/data /app/logs && \
    chown -R cerebro:cerebro /app
USER cerebro

ENTRYPOINT ["python", "-m", "cerebro.cli"]
CMD ["train", "--help"]