import os
import modal

image = modal.Image.debian_slim().pip_install(
    "huggingface_hub",
)

app = modal.App("finance-download-qwen", image=image)

hf_cache_vol = modal.Volume.from_name("hf-cache")

@app.function(
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    timeout=3600,
)
def download_model():
    from huggingface_hub import snapshot_download

    print("downloading Qwen model to hf-cache...")

    snapshot_download(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        cache_dir="/root/.cache/huggingface",
    )

    print("download finished.")
    hf_cache_vol.commit()

@app.local_entrypoint()
def main():
    download_model.remote()