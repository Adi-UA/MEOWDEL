from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as apply_spectral_norm


class Generator(nn.Module):
    """Maps a latent vector to a 3x64x64 image in [-1, 1] (Tanh output)."""

    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = 512

        self.project = nn.Sequential(
            nn.Linear(latent_dim, self.base_channels * 4 * 4),
            nn.BatchNorm1d(self.base_channels * 4 * 4),
            nn.LeakyReLU(0.2),
        )

        # 4 -> 8 -> 16 -> 32 -> 64, halving channels each stage
        channels = [512, 256, 128, 64, 32]
        stages = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            stages.append(self._upsample_block(in_ch, out_ch))
        self.upsample_stages = nn.Sequential(*stages)

        self.output = nn.Sequential(
            nn.Conv2d(channels[-1], 3, kernel_size=3, padding="same"),
            nn.Tanh(),
        )

    @staticmethod
    def _upsample_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, z):
        x = self.project(z)
        x = x.reshape(-1, self.base_channels, 4, 4)
        x = self.upsample_stages(x)
        return self.output(x)


class Discriminator(nn.Module):
    """Maps a 3x64x64 image to a single real/fake logit."""

    def __init__(self, spectral_norm=False):
        super().__init__()
        self.spectral_norm = spectral_norm

        # 64 -> 32 -> 16 -> 8 -> 4, growing channels each stage
        channels = [3, 64, 128, 256, 512]
        stages = []
        for i, (in_ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
            # No BatchNorm on the first layer (DCGAN guidance), and BatchNorm
            # is skipped entirely when spectral_norm is on since the two are
            # known to fight each other (SN-GAN, Miyato et al. 2018).
            use_bn = i > 0 and not spectral_norm
            stages.append(self._conv_block(in_ch, out_ch, use_bn))
        self.conv_stages = nn.Sequential(*stages)

        final_conv = nn.Conv2d(channels[-1], 1, kernel_size=4, stride=1, padding=0)
        if spectral_norm:
            final_conv = apply_spectral_norm(final_conv)
        self.output = nn.Sequential(final_conv, nn.Flatten())

    def _conv_block(self, in_channels, out_channels, use_bn):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        if self.spectral_norm:
            conv = apply_spectral_norm(conv)

        layers = [conv]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_stages(x)
        return self.output(x)
