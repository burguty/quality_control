from sentence_transformers import SentenceTransformer
import torch


class TextTower:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def __call__(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False
        ) # (B, TEXT_DIM)
