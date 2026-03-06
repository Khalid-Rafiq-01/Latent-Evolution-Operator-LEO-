import torch
import torch.nn as nn
import torch.nn.functional as F

class Norm(nn.Module):
    def __init__(self, num_channels, num_groups=4):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        return self.norm(x)


class EncoderVAE(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=2, padding=0),  # 96x192 -> 48x96
            nn.GELU(), Norm(32),

            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),  # 48x96 -> 24x48
            nn.GELU(), Norm(64),

            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0),  # 24x48 -> 12x24
            nn.GELU(), Norm(128),

            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),  # 12x24 -> 6x12
            nn.GELU(), Norm(256),

            nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=0),  # 6x12 -> 3x6
            nn.GELU(), Norm(512),
        )
        self.flatten = nn.Flatten()                    # 512 * 3 * 6 = 9216
        self.fc_mu     = nn.Linear(512 * 3 * 6, latent_dim)
        self.fc_logvar = nn.Linear(512 * 3 * 6, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        mu     = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 3 * 6)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.GELU(), Norm(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GELU(), Norm(256),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GELU(), Norm(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GELU(), Norm(128),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GELU(), Norm(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(), Norm(64),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GELU(), Norm(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(), Norm(32),

            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 3, 6)
        return self.deconv(x)


class PropagatorConcat(nn.Module):
    def __init__(self, latent_dim, feats=[16, 32, 64, 32, 16]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, feats[0]),
            nn.GELU(),
            nn.Linear(feats[0], feats[1]),
            nn.GELU(),
            nn.Linear(feats[1], feats[2]),
            nn.GELU(),
            nn.Linear(feats[2], feats[3]),
            nn.GELU(),
            nn.Linear(feats[3], feats[4]),
            nn.GELU(),
            nn.Linear(feats[4], latent_dim),
        )

    def forward(self, z, tau_norm):
        if tau_norm.dim() == 1:
            tau_norm = tau_norm.unsqueeze(1)
        z_in = torch.cat([z, tau_norm], dim=1)
        return self.net(z_in)


class VAE_LEO(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.encoder    = EncoderVAE(latent_dim)
        self.decoder    = Decoder(latent_dim)
        self.propagator = PropagatorConcat(latent_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, tau_norm):
        mu, logvar = self.encoder(x)            # (B, L), (B, L)
        z     = self.reparameterize(mu, logvar) # sampled latent
        z_tau = self.propagator(z, tau_norm)    # latent at t+τ

        x_hat     = self.decoder(z)
        x_hat_tau = self.decoder(z_tau)

        return x_hat, x_hat_tau, mu, logvar, z, z_tau

def beta_vae_kld(mu, logvar):
    # mean over batch and latent dims
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
