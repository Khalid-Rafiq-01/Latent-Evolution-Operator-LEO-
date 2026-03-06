import math
import torch
import numpy as np
import torch.nn as nn
from model_ldon_2d import *

# Code to load the data.
def load_latent_deeponet_advdiff_npz(npz_path, as_torch=False, device="cpu"):
    data = np.load(npz_path)

    branch = data["branch_inputs"]
    trunk  = data["trunk_inputs"]
    targets = data["targets"]
    alpha  = data["alpha_values"]

    print(f"Loaded: {npz_path}")
    print("branch_inputs:", branch.shape, branch.dtype)
    print("trunk_inputs :", trunk.shape, trunk.dtype)
    print("targets      :", targets.shape, targets.dtype)
    print("alpha_values :", alpha.shape, alpha.dtype)

    if as_torch:
        import torch
        branch_t  = torch.from_numpy(branch).float().to(device)
        trunk_t   = torch.from_numpy(trunk).float().to(device)
        targets_t = torch.from_numpy(targets).float().to(device)
        alpha_t   = torch.from_numpy(alpha).float().to(device)
        return branch_t, trunk_t, targets_t, alpha_t

    return branch, trunk, targets, alpha


def exact_solution(alpha, t, L=2.0, Nx=128, Ny=128, c=1.0):
    nu = 1.0 / alpha
    x_vals = np.linspace(-L, L, Nx)
    y_vals = np.linspace(-L, L, Ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    if t <= 0:
        return np.zeros_like(X)

    rx = X - c * t
    ry = Y
    r2 = rx**2 + ry**2

    denom = 4.0 * nu * t
    amp = 1.0 / (4.0 * np.pi * nu * t)
    U = amp * np.exp(-r2 / denom)
    return U.astype(np.float32)  # (Ny, Nx)


class BranchMLP(nn.Module):
    def __init__(self, in_dim, m, p, hidden=(128, 256), act=nn.ReLU):
        super().__init__()
        layers = []
        dims = (in_dim,) + hidden + (m * p,)
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]), act()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
        self.m, self.p = m, p

    def forward(self, x):                  # [B, in_dim]
        y = self.net(x)                    # [B, m*p]
        return y.view(-1, self.m, self.p)  # [B, m, p]


# Trunk Net: [B, 1] -> [B, m, p]
class TrunkMLP(nn.Module):
    def __init__(self, loc_dim, m, p, hidden=(100, 100), act=nn.ReLU):
        super().__init__()
        layers = []
        dims = (loc_dim,) + hidden + (m * p,)
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]), act()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
        self.m, self.p = m, p

    def forward(self, q):                  # [B, loc_dim]
        y = self.net(q)                    # [B, m*p]
        return y.view(-1, self.m, self.p)  # [B, m, p]


def normalize_z(z, mu, std, eps=1e-8):
    return (z - mu) / (std + eps)

def denormalize_z(z_norm, mu, std, eps=1e-8):
    return z_norm * (std + eps) + mu

def forward_latent(x, t, mu_z, std_z):
    """
    x: [B,1,H,W]
    t: [B,1]
    returns: z_pred_norm [B,m]
    """
    with torch.no_grad():
        z0 = encoder(x)                      # [B,m]
        z0 = normalize_z(z0, mu_z, std_z)    # [B,m] normalized

    yb = branch(z0)          # [B,m,p]
    yt = trunk(t)            # [B,m,p]

    z_pred_norm = torch.einsum("bmp,bmp->bm", yb, yt)  # [B,m]
    return z_pred_norm