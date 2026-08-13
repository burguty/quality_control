import torch
from sentence_transformers import SentenceTransformer


class TextTower:
    def __init__(self, encoder: SentenceTransformer):
        self.encoder = encoder

    @torch.inference_mode()
    def __call__(self, texts: list[str], prompt: str | None = None, batch_size: int = 32) -> torch.Tensor:
        return self.encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
            prompt=prompt,
        )
