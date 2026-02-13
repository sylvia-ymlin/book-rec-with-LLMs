# Server Deployment Guide (AutoDL)

This guide documents the specific steps required to deploy the Book Recommender system on an AutoDL (or similar domestic GPU cloud) server.

## 1. Environment Setup

The default environment on some cloud images may be outdated. Always create a fresh Conda environment.

```bash
# Create a fresh environment (Python 3.10 recommended)
conda create -n valid python=3.10 -y
conda activate valid

# Install dependencies
# Note: Use official PyPI to avoid stale mirrors returning ancient packages (like huggingface-hub 1.2.4)
pip install -r requirements.txt -i https://pypi.org/simple
```

**Critical Dependencies**:
- `huggingface-hub >= 0.23.0` (Required for modern transformers compatibility)
- `redis` (Python client)

## 2. Infrastructure Services

### Redis (Caching)
Ensure Redis Server is installed and running:
```bash
apt update && apt install redis-server -y
service redis-server start
```

## 3. Data Migration (Efficiently)

Do **NOT** upload the raw `Books_rating.csv` (2.7 GB) or uncompressed text files. Bandwidth is precious.

**Local Machine**:
```bash
# Compress large files
gzip -k data/books_processed.csv       # Metadata for API
gzip -k data/books_descriptions.txt    # Text for Vector DB

# Upload compressed files
scp data/books_processed.csv.gz root@<IP>:<PORT>:~/autodl-tmp/book-rec-with-LLMs/data/
scp data/books_descriptions.txt.gz root@<IP>:<PORT>:~/autodl-tmp/book-rec-with-LLMs/data/
```

**Server**:
```bash
# Decompress
gunzip -f data/*.gz
```

## 4. Model Downloading (Network Fix)

Domestic servers often cannot access Hugging Face directly. Use the official mirror.

**Server**:
```bash
# Enable Mirror
export HF_ENDPOINT=https://hf-mirror.com
# Increase Timeout for large files
export HF_HUB_DOWNLOAD_TIMEOUT=120

# Run Initialization (Downloads model + Builds Index)
python data/scripts/init_db.py
```

## 5. Running the Application

**Server**:
```bash
# Listen on 0.0.0.0 (required for external access)
uvicorn src.app.main:app --host 0.0.0.0 --port 6006
```

**Local Machine (Access)**:
Use SSH Tunneling to securely access the remote API without exposing ports publicly.
```bash
ssh -L 6006:localhost:6006 root@<IP> -p <PORT>
```
Visit `http://localhost:6006/docs` in your browser.
