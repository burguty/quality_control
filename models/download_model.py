from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoProcessor


DEFAULT_REPO_ID = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_MODELS_DIR = Path.home() / "quality_control" / "models"


def get_or_download_model(
    repo_id: str = DEFAULT_REPO_ID,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
):
    """
    Скачивает модель при необходимости, затем возвращает:
        model, processor

    Модель загружается через device_map='auto':
    Transformers/Accelerate самостоятельно размещают её на GPU/CPU.
    """
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
        print(f"[models] Скачиваю: {repo_id}")
        print(f"[models] В каталог: {local_model_dir}")

        local_model_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_model_dir),
        )
    else:
        print(f"[models] Использую локальную копию: {local_model_dir}")

    processor = AutoProcessor.from_pretrained(
        str(local_model_dir),
        local_files_only=True,
        trust_remote_code=True,
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModel.from_pretrained(
        str(local_model_dir),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    ).eval()

    return model, processor
