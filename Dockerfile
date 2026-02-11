FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "uvicorn[standard]"

# Copy application code
COPY . .

# Create a non-root user (Standard for HF Spaces)
RUN useradd -m -u 1000 user

# Create data directory and set permissions
# This ensures local fallback works and persistent storage mountpoint is accessible
RUN mkdir -p /app/data && chown -R user:user /app

# Switch to non-root user
USER user

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HOME=/home/user
ENV PATH=$HOME/.local/bin:$PATH

# Expose port for API
EXPOSE 8000

# Default command: Run FastAPI backend
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
