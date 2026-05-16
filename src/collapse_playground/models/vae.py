"""
Variational Autoencoder for MNIST/Fashion-MNIST.

Small architecture optimized for fast training on commodity hardware.
The model is intentionally compact — we care about generation behavior
across multiple training generations, not state-of-the-art sample quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Convolutional VAE for 28x28 grayscale images.

    Encoder: 28x28 -> conv -> conv -> flatten -> mu, logvar (latent_dim each)
    Decoder: latent_dim -> linear -> deconv -> deconv -> 28x28
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 1x28x28 -> 32x14x14 -> 64x7x7
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.enc_fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder: latent_dim -> 64x7x7 -> 32x14x14 -> 1x28x28
        self.dec_fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.dec_deconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.dec_deconv2 = nn.ConvTranspose2d(
            32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map input image to (mu, logvar) of latent distribution."""
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = h.flatten(start_dim=1)
        mu = self.enc_fc_mu(h)
        logvar = self.enc_fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample z = mu + sigma * eps, where eps ~ N(0, I).

        This makes the sampling differentiable so we can backprop through it.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent vector back to image space."""
        h = self.dec_fc(z)
        h = h.view(-1, 64, 7, 7)
        h = F.relu(self.dec_deconv1(h))
        x_recon = torch.sigmoid(self.dec_deconv2(h))
        return x_recon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Generate n new samples by sampling from prior N(0, I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standard VAE loss: reconstruction (BCE) + KL divergence to N(0, I).

    Args:
        x_recon: reconstructed images, shape (B, 1, 28, 28), in [0, 1]
        x: original images, same shape, in [0, 1]
        mu, logvar: encoder outputs, shape (B, latent_dim)
        beta: weight on KL term (beta-VAE; beta=1 is standard VAE)

    Returns:
        total loss, reconstruction loss, kl loss (all summed over batch)
    """
    # Reconstruction: pixel-wise binary cross entropy
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")

    # KL divergence: closed form for two Gaussians (q(z|x) || N(0, I))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss