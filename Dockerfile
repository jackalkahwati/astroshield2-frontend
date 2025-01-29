FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only the backend directory
COPY backend/ /app/

# Debug: List contents
RUN ls -la /app

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV PYTHON_MODULE=app.main
ENV APP_MODULE=${PYTHON_MODULE}:app

# Debug: Show Python path and installed packages
RUN python -c "import sys; print('Python path:', sys.path)" && \
    pip list

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start command with explicit Python path
CMD python -c "import sys; print('Starting with Python path:', sys.path)" && \
    python -m uvicorn ${APP_MODULE} --host 0.0.0.0 --port ${PORT} --workers 4 --log-level debug 