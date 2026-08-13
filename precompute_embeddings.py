from pathlib import Path
from tqdm import tqdm

import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from models.download_model import get_or_download_model
from library.model import Model
from library.dataset import OzonDataset


DEFAULT_DATA_PATH = Path("train_dataset/data.csv")
DEFAULT_IMAGES_PATH = Path("train_dataset/images/")
DEFAULT_EMBEDDINGS_PATH = Path("train_dataset/embeddings.pt")


@torch.inference_mode()
def precompute_embeddings(
    model: Model,
    dataset: OzonDataset,
    output_path: str | Path,
):
    text_embeds = []
    image_embeds = []
    categories = []
    labels = []

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=OzonDataset.collate_fn,
    )

    for batch in tqdm(loader, desc="Precomputing embeddings"):
        text_embed = model.get_text_embeds(batch)
        image_embed = model.get_image_embeds(batch)

        text_embeds.append(text_embed.cpu())
        image_embeds.append(image_embed.cpu())

        categories.extend(batch["category"])
        labels.append(batch["label"].cpu())

    data = {
        "text_embed": torch.cat(text_embeds, dim=0),
        "image_embed": torch.cat(image_embeds, dim=0),
        "category": categories,
        "label": torch.cat(labels, dim=0),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(data, output_path)

    print(f"Saved embeddings to: {output_path}")
    print(f"text_embed:  {data['text_embed'].shape}")
    print(f"image_embed: {data['image_embed'].shape}")
    print(f"label:       {data['label'].shape}")
    print(f"category:    {len(data['category'])}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    image_transform = v2.Compose([
        v2.Resize(size=672, antialias=True),
    ])
    dataset = OzonDataset(DEFAULT_DATA_PATH, DEFAULT_IMAGES_PATH, image_transform)
    encoder = get_or_download_model(device=device)
    model = Model(encoder, hidden_dim=1, n_hidden_layers=1).to(device)

    precompute_embeddings(model, dataset, output_path=DEFAULT_EMBEDDINGS_PATH)


if __name__ == "__main__":
    main()
