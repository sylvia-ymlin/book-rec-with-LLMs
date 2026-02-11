import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

print("🚀 Downloading model from hf-mirror...")
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    ignore_patterns=["*.bin", "*.h5", "*.ot"],  # 只下载 safetensors，省流
    resume_download=True
)
print("✅ Download Complete!")
