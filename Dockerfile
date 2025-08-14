# DDR5 AI Sandbox Simulator Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for hardware detection
# System dependencies for hardware detection and common ML/visualization libs
# Also install curl for container healthcheck and build tooling for wheels when needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    dmidecode \
    curl \
    ca-certificates \
    git \
    build-essential \
    cmake \
    pkg-config \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Optional: allow preinstall of PyTorch CPU wheels to ensure availability in CI
# Provide these via build-args; defaults are empty (skips this step)
ARG TORCH_INDEX_URL=
ARG TORCH_VERSION=
ARG TORCHVISION_VERSION=
ARG TORCHAUDIO_VERSION=

# Install Python dependencies
# If TORCH_INDEX_URL and versions are provided, install torch stack first to avoid resolution issues
RUN if [ -n "$TORCH_INDEX_URL" ] && [ -n "$TORCH_VERSION" ]; then \
            pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" \
                torch=="$TORCH_VERSION" \
                ${TORCHVISION_VERSION:+torchvision=="$TORCHVISION_VERSION"} \
                ${TORCHAUDIO_VERSION:+torchaudio=="$TORCHAUDIO_VERSION"} ; \
        else \
            echo "Skipping preinstall of torch (no TORCH_INDEX_URL/versions provided)" ; \
        fi && \
        pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 simulator && \
    chown -R simulator:simulator /app
USER simulator

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -sf http://localhost:8501/_stcore/health || exit 1

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
