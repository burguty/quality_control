from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import f1_score, precision_score, recall_score

from models.download_model import get_or_download_model
from library.model import Model
from library.dataset import OzonDataset


DEFAULT_DATA_PATH = Path("train_dataset/data.csv")
DEFAULT_IMAGES_PATH = Path("train_dataset/images/")


def load_dataset(
    batch_size: int = 4,
    valid_size: float = 0.2,
    seed: int = 42,
):
    dataset = OzonDataset(path_to_data=DEFAULT_DATA_PATH, path_to_images=DEFAULT_IMAGES_PATH)

    valid_size = int(len(dataset) * valid_size)
    train_size = len(dataset) - valid_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=OzonDataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=OzonDataset.collate_fn)

    return train_loader, valid_loader


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        labels = batch["label"].to(device)

        logits = model(batch)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)

        preds = (torch.sigmoid(logits) >= 0.5).long()

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.long().cpu().tolist())

    loss = total_loss / len(loader.dataset)

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "loss": loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()

    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        labels = batch["label"].to(device)

        logits = model(batch)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epoch", leave=False):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = validate(model, valid_loader, criterion, device)

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} | "
            f"valid_loss={metrics['loss']:.4f} | "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"f1={metrics['f1']:.4f}"
        )


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # DATA
    train_loader, valid_loader = load_dataset(batch_size=4)

    # MODEL INITIALIZATION
    encoder = get_or_download_model(device=device)
    model = Model(encoder, hidden_dim=512, n_hidden_layers=2)
    model = model.to(device)

    # TRAINING
    train_model(model, train_loader, valid_loader, device, num_epochs=10)


if __name__ == "__main__":
    main()
