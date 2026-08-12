from PIL import Image
from sentence_transformers import SentenceTransformer
import torch


class ImageTower:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def __call__(self, images: list[Image.Image], batch_size: int = 8) -> torch.Tensor:
        return self.model.encode(
            images,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )  # (B, IMAGE_DIM)
