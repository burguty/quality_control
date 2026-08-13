import torch
from sentence_transformers import SentenceTransformer


class ImageTower:
    def __init__(self, encoder: SentenceTransformer):
        self.encoder = encoder

    @torch.inference_mode()
    def __call__(self, images: list[torch.Tensor], prompt: str | None = None, batch_size: int = 1) -> torch.Tensor:
        embeddings = self.encoder.encode(
            images,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
            prompt=prompt,
        )

        return embeddings.unsqueeze(0) if embeddings.ndim == 1 else embeddings
