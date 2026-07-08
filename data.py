import os
from pathlib import Path

import kagglehub
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class FlatImageDataset(Dataset):
    """Loads every image file found recursively under `root`, ignoring any
    class-subfolder structure (the Kaggle cat faces download is a flat dump
    of images, not an ImageFolder-style layout)."""

    def __init__(self, root, transform=None):
        self.paths = sorted(
            p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0


def get_dataloader(config):
    """Returns a single training dataloader over the full dataset.

    GANs don't have a principled validation loss (there's no ground truth to
    score reconstructions against), so there's no train/val split here -
    sample quality is checked qualitatively via the generated image grids.
    """
    dataset_root = kagglehub.dataset_download(config["kaggle_dataset"])

    image_size = config["image_size"]
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = FlatImageDataset(dataset_root, transform=transform)

    cpu_count = os.cpu_count() or 1
    num_workers = max(1, round(cpu_count * 0.6 / 2) * 2)

    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
