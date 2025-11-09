# Quantum Bioenergetics Mapping Platform - Dockerfile
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash qbm

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application directories
RUN mkdir -p /app /app/results /app/temp /app/data && \
    chown -R qbm:qbm /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=qbm:qbm . .

# Copy sample data
RUN cp -r data/* /app/data/ || true

# Switch to non-root user
USER qbm

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose ports
EXPOSE 8501 8000

# Default command - run Streamlit app
CMD ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Development override
# To run in development mode with API server:
# docker run -p 8501:8501 -p 8000:8000 qbm-platform bash -c "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run app/Home.py --server.port 8501"
