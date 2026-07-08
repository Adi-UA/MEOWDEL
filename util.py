import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def make_run_dir(base="outputs"):
    run_dir = Path(base) / datetime.now().strftime("%Y%m%d_%H%M%S")
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_image_grid(images, path, nrow=3):
    """images: tensor of shape (N, C, H, W) in [-1, 1]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    images = (images.clamp(-1, 1) + 1) / 2
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    grid = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(nrow * 2, nrow * 2))
    plt.imshow(grid)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_losses(history, path):
    """history: dict of metric name -> list of per-epoch values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for name, values in history.items():
        plt.plot(values, marker=".", label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, map_location=None):
    model.load_state_dict(torch.load(path, map_location=map_location))
    return model
