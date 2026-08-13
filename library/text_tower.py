from sentence_transformers import SentenceTransformer
import torch


class TextTower:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def __call__(self, texts: list[str], batch_size: int = 32, output_value: str = 'sentence_embedding') -> torch.Tensor:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            output_value=output_value,
        )
