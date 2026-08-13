from pathlib import Path

import pandas as pd

import torch
from torchvision import io
from torch.utils.data import Dataset


class OzonDataset(Dataset):
    def __init__(
        self,
        path_to_data: str = "./train_dataset/data.csv",
        path_to_images: str = "./train_dataset/images/",
    ):
        super().__init__()

        self.df = pd.read_csv(path_to_data)
        self.images_dir = Path(path_to_images)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        product_id = int(row["id"])
        image_dir = self.images_dir / str(product_id)

        images = []

        if image_dir.exists():
            for path in sorted(image_dir.iterdir()):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    image = io.read_image(path, mode=io.ImageReadMode.RGB)
                    images.append(image)

        return {
            "id": product_id,
            "name": str(row["name"]),
            "category": str(row["category"]),
            "description": str(row["description"]),
            "images": images,
            "images_count": len(images),
            "label": int(row["label"]),
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "id": [item["id"] for item in batch],
            "name": [item["name"] for item in batch],
            "category": [item["category"] for item in batch],
            "description": [item["description"] for item in batch],
            "images": [item["images"] for item in batch],
            "images_count": [item["images_count"] for item in batch],
            "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        }
