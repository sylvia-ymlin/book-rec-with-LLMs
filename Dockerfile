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

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose ports for both API and Gradio
EXPOSE 8000
EXPOSE 7860

# Default command: Run the API
# You can override this to run "python app.py" for the UI
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
