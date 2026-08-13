from typing import Callable, Dict

import torch
from torch import nn
from sentence_transformers import SentenceTransformer

from library.image_tower import ImageTower
from library.text_tower import TextTower


class Model(nn.Module):
    def __init__(
        self,
        encoder: SentenceTransformer,
        hidden_dim: int,
        n_hidden_layers: int,
        activation: Callable = nn.ReLU,
    ):
        super().__init__()

        self.encoder = encoder
        self.image_tower = ImageTower(encoder)
        self.text_tower = TextTower(encoder)

        embedding_dim = encoder.get_embedding_dimension()

        layers = []
        in_dim = 2 * embedding_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_text_embeds(self, inputs: Dict) -> torch.Tensor:
        texts = [
            (
                f"Название: {name}\n"
                f"Описание: {description}\n"
                f"Категория: {category}"
            )
            for name, description, category in zip(
                inputs["name"],
                inputs["description"],
                inputs["category"],
            )
        ]

        out = self.text_tower(texts)
        return out

    def get_image_embeds(self, inputs: Dict) -> torch.Tensor:
        flat_images = [image for product_images in inputs["images"] for image in product_images]
        counts = inputs["images_count"]

        flat_embeds = self.image_tower(flat_images) # (total_images, image_dim)

        product_embeds = []
        offset = 0

        for count in counts:
            embeds = flat_embeds[offset : offset + count]
            product_embeds.append(embeds.mean(dim=0))
            offset += count

        return torch.stack(product_embeds, dim=0)

    def forward(self, inputs: Dict) -> torch.Tensor:
        text_embeds = self.get_text_embeds(inputs).to(self.device)
        image_embeds = self.get_image_embeds(inputs).to(self.device)

        x = torch.cat([text_embeds, image_embeds], dim=-1)
        x = self.net(x)
        x = x.squeeze(x)
        return x
