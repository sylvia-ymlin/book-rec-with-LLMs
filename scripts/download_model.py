import os
import argparse

def download_model(repo_id, local_dir=None):
    """
    Downloads a model from HuggingFace Mirror (for China/Restricted Networks).
    """
    # Force use of mirror
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    
    from huggingface_hub import snapshot_download

    print(f"🚀 Downloading {repo_id} from hf-mirror.com...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=["*.bin", "*.h5", "*.ot", "*.msgpack"],  # Prefer safetensors
        resume_download=True
    )
    print("✅ Download Complete!")

if __name__ == "__main__":
    download_model("sentence-transformers/all-MiniLM-L6-v2")
