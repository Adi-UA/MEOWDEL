import pytest
import torch
from PIL import Image
from torchvision import transforms

from data import FlatImageDataset

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def _write_fake_images(root, count):
    for i in range(count):
        Image.new("RGB", (80, 80), color=(i * 10, i * 20, i * 30)).save(root / f"cat_{i}.jpg")


def test_finds_images_recursively(tmp_path):
    (tmp_path / "nested").mkdir()
    _write_fake_images(tmp_path, 2)
    _write_fake_images(tmp_path / "nested", 3)

    dataset = FlatImageDataset(tmp_path, transform=TRANSFORM)
    assert len(dataset) == 5


def test_getitem_returns_transformed_tensor_and_label(tmp_path):
    _write_fake_images(tmp_path, 1)

    dataset = FlatImageDataset(tmp_path, transform=TRANSFORM)
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 64, 64)
    assert label == 0


def test_raises_on_empty_directory(tmp_path):
    with pytest.raises(RuntimeError):
        FlatImageDataset(tmp_path, transform=TRANSFORM)
