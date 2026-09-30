# =============================================================================
# Dockerfile — RAG-Based AI Teaching Assistant
# =============================================================================
# Production multi-platform Docker container running Streamlit on CPU.
# Optimized for lightweight deployment with pre-installed FFmpeg for media decoding.
# =============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    PORT=8501

WORKDIR /app

# Install system dependencies:
# - ffmpeg: Required by Whisper for audio extraction from lecture videos
# - curl: Used by container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python production dependencies first for efficient layer caching
COPY requirements-prod.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# Copy application configuration, source code, data, and scripts
COPY .streamlit/ /app/.streamlit/
COPY src/ /app/src/
COPY data/ /app/data/
COPY scripts/ /app/scripts/
COPY app.py /app/
COPY .env.example /app/

# Create non-root user for security hygiene
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose default Streamlit port
EXPOSE 8501

# Container healthcheck querying Streamlit's internal health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit web application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
