import torch

from model import Generator
from util import load_model, make_run_dir, save_model


def test_make_run_dir_creates_expected_structure(tmp_path):
    run_dir = make_run_dir(base=tmp_path)
    assert run_dir.parent == tmp_path
    assert (run_dir / "samples").is_dir()
    assert (run_dir / "checkpoints").is_dir()


def test_save_and_load_model_roundtrip(tmp_path):
    original = Generator(latent_dim=8)
    path = tmp_path / "generator.pth"
    save_model(original, path)

    loaded = Generator(latent_dim=8)
    load_model(loaded, path)

    for (name, original_param), (_, loaded_param) in zip(
        original.state_dict().items(), loaded.state_dict().items()
    ):
        assert torch.equal(original_param, loaded_param), f"{name} did not round-trip"
