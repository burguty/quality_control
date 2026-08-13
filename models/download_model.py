from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer


DEFAULT_REPO_ID = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_MODELS_DIR = Path.home() / "quality_control" / "models"
DEFAULT_EMBEDDING_DIM = 2048


def get_or_download_model(
    repo_id: str = DEFAULT_REPO_ID,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    device: torch.device | str = "cpu",
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> SentenceTransformer:
    models_dir = Path(models_dir)
    local_model_dir = models_dir / repo_id

    has_config = (local_model_dir / "config.json").is_file()
    has_weights = (
        (local_model_dir / "model.safetensors").is_file()
        or (local_model_dir / "model.safetensors.index.json").is_file()
        or (local_model_dir / "pytorch_model.bin").is_file()
        or (local_model_dir / "pytorch_model.bin.index.json").is_file()
    )

    if not (has_config and has_weights):
        print(f"[models] Downloading: {repo_id}")
        print(f"[models] Destination: {local_model_dir}")

        local_model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=str(local_model_dir))

    else:
        print(f"[models] Using local copy: {local_model_dir}")

    model = SentenceTransformer(str(local_model_dir), device=device, truncate_dim=embedding_dim)
    return model
