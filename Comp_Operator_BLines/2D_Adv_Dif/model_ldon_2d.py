import math
import torch
import numpy as np
import torch.nn as nn

# Normalization Layer for Conv2D
class Norm(nn.Module):
    def __init__(self, num_channels, num_groups=4):
        super(Norm, self).__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        return self.norm(x)

# Encoder using Conv2D
class Encoder(nn.Module):
    def __init__(self, latent_dim=3):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            # Input: (batch_size, 1, 256, 256)
            nn.Conv2d(1, 32, kernel_size=2, stride=2, padding=0),  # (batch_size, 64, 128, 128)
            nn.GELU(),
            Norm(32),
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0),  # (batch_size, 128, 64, 64)
            nn.GELU(),
            Norm(64),
            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0),  # (batch_size, 256, 32, 32)
            nn.GELU(),
            Norm(128),
            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),  # (batch_size, 512, 16, 16)
            nn.GELU(),
            Norm(256),
            nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=0),  # (batch_size, 512, 8, 8)
            nn.GELU(),
            Norm(512),
        )
        
        self.flatten = nn.Flatten()
        self.latent = nn.Linear(512 * 4 * 4, latent_dim)
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        z = self.latent(x)
        return z
        
        
        
class Decoder(nn.Module):
    def __init__(self, latent_dim=3):
        super(Decoder, self).__init__()
        # Fully connected layer to transform the latent vector back to the shape (batch_size, 512, 8, 8)
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        self.deconv_layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.GELU(),
            Norm(256),


            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.GELU(),
            Norm(128),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.GELU(),
            Norm(64),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.GELU(),
            Norm(32),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, z):
        # Transform the latent vector to match the shape of the feature maps
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)  # Reshape to (batch_size, 512, 4, 4)
        x = self.deconv_layers(x)
        return x
        
        
class AE_2D_AdvDif(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE_2D_AdvDif, self).__init__()
        self.encoder = encoder
        self.decoder = decoder # decoder for x(t)

    def forward(self, x):
        
        z = self.encoder(x)
        # Reconstruction
        x_hat = self.decoder(z)  # Reconstruction of x(t)
        return x_hat, z