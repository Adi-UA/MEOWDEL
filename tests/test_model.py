import torch

from model import Discriminator, Generator


def test_generator_output_shape():
    generator = Generator(latent_dim=128)
    z = torch.randn(4, 128)
    out = generator(z)
    assert out.shape == (4, 3, 64, 64)


def test_generator_output_range():
    generator = Generator(latent_dim=128)
    out = generator(torch.randn(2, 128))
    assert out.min() >= -1.0 and out.max() <= 1.0  # Tanh output


def test_discriminator_output_shape():
    discriminator = Discriminator(spectral_norm=False)
    x = torch.randn(4, 3, 64, 64)
    out = discriminator(x)
    assert out.shape == (4, 1)


def test_discriminator_with_spectral_norm():
    discriminator = Discriminator(spectral_norm=True)
    x = torch.randn(4, 3, 64, 64)
    out = discriminator(x)
    assert out.shape == (4, 1)


def test_generator_discriminator_backward_pass_no_nan():
    generator = Generator(latent_dim=128)
    discriminator = Discriminator(spectral_norm=False)

    z = torch.randn(4, 128)
    fake = generator(z)
    out = discriminator(fake)
    loss = out.sum()
    loss.backward()

    for name, param in list(generator.named_parameters()) + list(discriminator.named_parameters()):
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"
