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
python src/init_db.py
```

## 5. Running the Application

**Server**:
```bash
# Listen on 0.0.0.0 (required for external access)
uvicorn src.main:app --host 0.0.0.0 --port 6006
```

**Local Machine (Access)**:
Use SSH Tunneling to securely access the remote API without exposing ports publicly.
```bash
ssh -L 6006:localhost:6006 root@<IP> -p <PORT>
```
Visit `http://localhost:6006/docs` in your browser.

## 图片兜底与路径适配说明

### 现象
- 书籍图片缺失时，前端 `<img src="/assets/cover-not-found.jpg">` 无法正常显示默认图片。
- 原因：开发环境下前端端口（如 5173）与后端端口（如 6006）不同，`/assets` 路径实际指向前端静态目录，无法访问后端 FastAPI 挂载的静态资源。

### 解决方案
- 后端 FastAPI 通过 `app.mount("/assets", StaticFiles(directory="assets"), name="assets")` 挂载静态资源。
- 前端图片加载失败时，自动切换为后端 API 地址的兜底图片：

```jsx
<img
  src={book.img}
  alt={book.title}
  onError={e => {
    e.target.onerror = null;
    e.target.src = "http://localhost:6006/assets/cover-not-found.jpg";
  }}
/>
```
- 这样无论图片链接是否有效，缺失时都能正常显示默认封面。

### 生产环境建议
- 生产部署时建议用 nginx 统一代理 `/assets` 到后端或静态目录，保证前后端一致。

---
