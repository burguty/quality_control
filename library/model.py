from typing import Callable

import torch
from torch import nn


class Model(nn.Module):
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        activation: Callable = nn.ReLU,
    ):
        super().__init__()

        layers = []
        in_dim = text_dim + image_dim

        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))

        self.feedforward = nn.Sequential(*layers)

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor) -> torch.Tensor:
        # text_embeds:  (B, text_dim)
        # image_embeds: (B, image_dim)

        x = torch.cat([text_embeds, image_embeds], dim=-1)
        x = self.feedforward(x)  # (B, 1)
        x = x.squeeze(-1)  # (B,)
        return x
