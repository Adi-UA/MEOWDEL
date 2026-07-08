import argparse
import csv
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from data import get_dataloader
from gan_module import LitGAN
from model import Discriminator, Generator
from util import make_run_dir, plot_losses, save_model, seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    torch.set_float32_matmul_precision("high")

    seed_everything(config["seed"])

    run_dir = make_run_dir(config["output_dir"])
    print(f"Run directory: {run_dir}")

    train_dataloader = get_dataloader(config)

    generator = Generator(latent_dim=config["latent_dim"])
    discriminator = Discriminator(spectral_norm=config["spectral_norm"])
    lit_gan = LitGAN(generator, discriminator, config, run_dir=run_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints", save_last=True, save_top_k=0
    )

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        devices=1,
        precision=config["precision"],
        logger=[
            TensorBoardLogger(save_dir=str(run_dir), name="", version=""),
            CSVLogger(save_dir=str(run_dir), name="", version=""),
        ],
        callbacks=[checkpoint_callback],
        log_every_n_steps=config["log_interval"],
    )

    trainer.fit(model=lit_gan, train_dataloaders=train_dataloader)

    save_model(generator, Path(config["save_path"]) / "generator.pth")
    save_model(discriminator, Path(config["save_path"]) / "discriminator.pth")
    print(f"Saved models to {config['save_path']}")

    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        history = {"g_loss": [], "d_loss": [], "r1_penalty": []}
        with open(metrics_path, "r") as f:
            for row in csv.DictReader(f):
                for key in history:
                    if row.get(key):
                        history[key].append(float(row[key]))
        history = {k: v for k, v in history.items() if v}
        plot_losses(history, run_dir / "loss_curve.png")
        print(f"Saved loss curve to {run_dir / 'loss_curve.png'}")
