import math
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F

from util import save_image_grid


class LitGAN(L.LightningModule):
    """Alternating GAN training (standard label convention: real=1, fake=0).

    Uses manual optimization since GAN training doesn't fit Lightning's
    single-optimizer automatic step: each training_step runs a full D update
    then a full G update, mirroring the classic alternating-SGD GAN algorithm
    step for step.
    """

    def __init__(self, generator, discriminator, config, run_dir=None):
        super().__init__()
        self.automatic_optimization = False
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.run_dir = Path(run_dir) if run_dir is not None else None

        self.register_buffer(
            "fixed_noise",
            torch.randn(config["sample_grid_size"], config["latent_dim"]),
        )

    def configure_optimizers(self):
        betas = (self.config["beta1"], self.config["beta2"])
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.config["lr_g"], betas=betas
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.config["lr_d"], betas=betas
        )
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx):
        real, _ = batch
        batch_size = real.shape[0]
        latent_dim = self.config["latent_dim"]
        opt_g, opt_d = self.optimizers()

        # --- Discriminator step ---
        z = torch.randn(batch_size, latent_dim, device=self.device)
        fake = self.generator(z).detach()

        r1_gamma = self.config.get("r1_gamma", 0.0)
        r1_interval = self.config.get("r1_interval", 1)
        apply_r1 = r1_gamma > 0 and self.global_step % r1_interval == 0
        if apply_r1:
            real = real.detach().requires_grad_(True)

        d_real = self.discriminator(real)
        d_fake = self.discriminator(fake)

        d_loss = F.binary_cross_entropy_with_logits(
            d_real, torch.ones_like(d_real)
        ) + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))

        r1_penalty = torch.zeros((), device=self.device)
        if apply_r1:
            grad_real = torch.autograd.grad(
                outputs=d_real.sum(), inputs=real, create_graph=True
            )[0]
            r1_penalty = grad_real.pow(2).reshape(batch_size, -1).sum(1).mean()
            # Lazy regularization (StyleGAN2): applied every r1_interval steps,
            # scaled up by r1_interval to keep the expected penalty magnitude
            # the same as applying a smaller penalty every step.
            d_loss = d_loss + (r1_gamma / 2) * r1_penalty * r1_interval

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        # --- Generator step ---
        z = torch.randn(batch_size, latent_dim, device=self.device)
        fake = self.generator(z)
        d_fake_for_g = self.discriminator(fake)
        g_loss = F.binary_cross_entropy_with_logits(
            d_fake_for_g, torch.ones_like(d_fake_for_g)
        )

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log_dict(
            {"g_loss": g_loss, "d_loss": d_loss, "r1_penalty": r1_penalty},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def on_train_epoch_end(self):
        if self.run_dir is None:
            return

        self.generator.eval()
        with torch.no_grad():
            samples = self.generator(self.fixed_noise)
        self.generator.train()

        nrow = int(math.isqrt(self.config["sample_grid_size"]))
        path = self.run_dir / "samples" / f"epoch_{self.current_epoch:03d}.png"
        save_image_grid(samples, path, nrow=nrow)
