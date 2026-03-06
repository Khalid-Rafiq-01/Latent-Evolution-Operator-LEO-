#!/usr/bin/env python
# coding: utf-8

# In[1]:

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
    def __init__(self, input_dim, latent_dim=2, feats=[512, 256, 128, 64, 32]):
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
    def __init__(self, latent_dim, output_dim, feats=[32, 64, 128, 256, 512]):
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

class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder # decoder for x(t)

    def forward(self, x):
        z = self.encoder(x)
        # Reconstruction
        x_hat = self.decoder(z)  # Reconstruction of x(t)

        return x_hat, z


# DeepONet heads
class BranchMLP(nn.Module):
    """Branch net: latent IC z0 (m-dim) → (m*p)."""
    def __init__(self, in_dim, m, p, hidden=(128, 256), act=nn.ReLU):
        super().__init__()
        layers = []
        dims = (in_dim,) + hidden + (m*p,)
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), act()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
        self.m, self.p = m, p
    def forward(self, x):              # x: [B, in_dim]
        y = self.net(x)                # [B, m*p]
        return y.view(-1, self.m, self.p)

class TrunkMLP(nn.Module):
    """Trunk net: normalized location q (1-dim) → (m*p)."""
    def __init__(self, loc_dim, m, p, hidden=(100, 100), act=nn.ReLU):
        super().__init__()
        layers = []
        dims = (loc_dim,) + hidden + (m*p,)
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), act()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
        self.m, self.p = m, p
    def forward(self, q):              # q: [B, loc_dim]
        y = self.net(q)                # [B, m*p]
        return y.view(-1, self.m, self.p)
