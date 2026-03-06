import torch
import torch.nn as nn

class Norm_new(nn.Module):
    def __init__(self, num_channels, num_groups=4):
        super(Norm_new, self).__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        if x.dim() == 2:
            # Reshape to (batch_size, num_channels, 1)
            x = x.unsqueeze(-1)
            x = self.norm(x)
            # Reshape back to (batch_size, num_channels)
            x = x.squeeze(-1)
        else:
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim=128, latent_dim=2, feats=[512, 256, 128, 64, 32]):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self._net = nn.Sequential(
            nn.Linear(input_dim, feats[0]),
            nn.GELU(),
            Norm_new(feats[0]),
            nn.Linear(feats[0], feats[1]),
            nn.GELU(),
            Norm_new(feats[1]),
            nn.Linear(feats[1], feats[2]),
            nn.GELU(),
            Norm_new(feats[2]),
            nn.Linear(feats[2], feats[3]),
            nn.GELU(),
            Norm_new(feats[3]),
            nn.Linear(feats[3], feats[4]),
            nn.GELU(),
            Norm_new(feats[4]),
            nn.Linear(feats[4], latent_dim)
        )

    def forward(self, x):
      Z = self._net(x)
      return Z

class Decoder(nn.Module):
    def __init__(self, latent_dim=2, output_dim=128, feats=[32, 64, 128, 256, 512]):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self._net = nn.Sequential(
            nn.Linear(latent_dim, feats[0]),
            nn.GELU(),
            Norm_new(feats[0]),
            nn.Linear(feats[0], feats[1]),
            nn.GELU(),
            Norm_new(feats[1]),
            nn.Linear(feats[1], feats[2]),
            nn.GELU(),
            Norm_new(feats[2]),
            nn.Linear(feats[2], feats[3]),
            nn.GELU(),
            Norm_new(feats[3]),
            nn.Linear(feats[3], feats[4]),
            nn.GELU(),
            Norm_new(feats[4]),
            nn.Linear(feats[4], output_dim),
            nn.Tanh()
        )

    def forward(self, x):
      y = self._net(x)
      return y

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder # decoder for x(t)

    def forward(self, x):
        z = self.encoder(x)

        # Reconstruction
        x_hat = self.decoder(z)  # Reconstruction of x(t)

        return x_hat, z

# X -> Enc -> z -> Dec -> X_hat
# X_tau -> Enc -> z_tau -> Dec -> X_hat_tau
# z_tau_prop = P(z_tau||re||tau)
# L = ||X, X_hat||, ||X_tau, X_hat_tau||, ||z_tau_prop, z_tau||

# Defining the Propagator model:
# I/P -> z|tau|Re O/p -> z_tau_hat
class Propagator(nn.Module):
    """(z_t, tau, Re) -> z_{t+tau}"""
    def __init__(self, latent_dim=2, feats=[16,32]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2, feats[0]), nn.ReLU(),
            nn.Linear(feats[0], feats[1]), nn.ReLU(),
            nn.Linear(feats[1], latent_dim),
        )
    def forward(self, z, tau, re):
        z_ = torch.cat([z, tau, re], dim=1)
        return self.net(z_)