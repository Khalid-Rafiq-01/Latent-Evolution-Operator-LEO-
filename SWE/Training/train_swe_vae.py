import os
import re
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb

from model_vae import *


class SWE_EvOpDataset(Dataset):
    def __init__(self, U, file_paths, max_tau, samples_per_traj,
                 u_mean=None, u_std=None):
        """
        U: numpy array (N_traj, T, H, W)
        file_paths: list of len N_traj with original nc paths
        max_tau: max time jump in steps (<= T-1)
        samples_per_traj: how many random pairs per trajectory per epoch
        u_mean, u_std: scalars for normalization
        """
        self.U = torch.from_numpy(U)  # (N, T, H, W), float32
        self.file_paths = file_paths
        self.N, self.T, self.H, self.W = self.U.shape
        self.max_tau = max_tau
        self.samples_per_traj = samples_per_traj

        if u_mean is None:
            u_mean = self.U.mean()
        if u_std is None:
            u_std = self.U.std()
        self.u_mean = float(u_mean)
        self.u_std = float(u_std)

    def __len__(self):
        return self.N * self.samples_per_traj

    def _parse_seed_run(self, traj_id):
        path = self.file_paths[traj_id]
        seed_match = re.search(r"seed=(\d+)", path)
        run_match  = re.search(r"run(\d+)", path)
        seed = int(seed_match.group(1)) if seed_match else -1
        run  = int(run_match.group(1))  if run_match  else -1
        return seed, run

    def __getitem__(self, idx):
        traj_id = idx // self.samples_per_traj

        t0 = torch.randint(0, self.T - 1, (1,)).item()

        tau_max_here = min(self.max_tau, self.T - 1 - t0)
        tau = torch.randint(1, tau_max_here + 1, (1,)).item()

        x_t     = self.U[traj_id, t0]
        x_t_tau = self.U[traj_id, t0 + tau]

        x_t     = (x_t - self.u_mean) / self.u_std
        x_t_tau = (x_t_tau - self.u_mean) / self.u_std

        x_t     = x_t.unsqueeze(0)
        x_t_tau = x_t_tau.unsqueeze(0)

        tau_norm = tau / (self.T - 1)

        seed, run = self._parse_seed_run(traj_id)

        return {
            "x_t": x_t,
            "x_t_tau": x_t_tau,
            "tau": torch.tensor(tau, dtype=torch.long),
            "tau_norm": torch.tensor(tau_norm, dtype=torch.float32),
            "traj_id": traj_id,
            "seed": seed,
            "run": run,
            "t0": t0,
        }


def load_swe_data(train_root="./train/", val_root="./valid/"):
    train_files = sorted(glob.glob(os.path.join(train_root, "seed=*/run*/output.nc")))
    print("num train files:", len(train_files))

    train_trajs = []
    for f in train_files:
        ds = xr.open_dataset(f)
        u = ds["u"].isel(lev=0).values.astype(np.float32)
        train_trajs.append(u)

    U_train = np.stack(train_trajs, axis=0)
    print("U_train shape:", U_train.shape)

    val_files = sorted(glob.glob(os.path.join(val_root, "seed=*/run*/output.nc")))
    print("num val files:", len(val_files))

    val_trajs = []
    for f in val_files:
        ds = xr.open_dataset(f)
        u = ds["u"].isel(lev=0).values.astype(np.float32)
        val_trajs.append(u)

    U_val = np.stack(val_trajs, axis=0)
    print("U_val shape:", U_val.shape)

    u_mean = float(U_train.mean())
    u_std  = float(U_train.std())
    print("u_mean:", u_mean, "u_std:", u_std)

    return U_train, U_val, train_files, val_files, u_mean, u_std


def make_dataloaders(
    U_train, U_val, train_files, val_files,
    u_mean, u_std,
    samples_per_traj_train,
    batch_size=16,
    max_tau=87,
    samples_per_traj_val=5,
    num_workers=4,
):
    train_dataset = SWE_EvOpDataset(
        U_train, train_files,
        max_tau=max_tau,
        samples_per_traj=samples_per_traj_train,
        u_mean=u_mean, u_std=u_std,
    )

    val_dataset = SWE_EvOpDataset(
        U_val, val_files,
        max_tau=max_tau,
        samples_per_traj=samples_per_traj_val,
        u_mean=u_mean, u_std=u_std,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_loader, val_loader


def train(config=None):
    default_config = dict(
        lr=3e-4,
        latent_dim=32,
        lambda_dir=3.0,
        beta=1e-5,
        epochs=50,
        samples_per_traj_train=10,
        project="d_leo_vae_swe",
        run_name=None,
    )

    with wandb.init(config=config, project=default_config["project"],
                    name=None if config is None else None) as run:
        cfg = wandb.config

        for k, v in default_config.items():
            if k not in cfg:
                setattr(cfg, k, v)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)

        U_train, U_val, train_files, val_files, u_mean, u_std = load_swe_data()
        train_loader, val_loader = make_dataloaders(
            U_train, U_val, train_files, val_files,
            u_mean, u_std,
            samples_per_traj_train=cfg.samples_per_traj_train,
            batch_size=16,
        )

        model = VAE_LEO(latent_dim=cfg.latent_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs
        )

        lambda_dir = cfg.lambda_dir
        beta       = cfg.beta
        num_epochs = cfg.epochs  # fixed 50 via default_config

        best_val_loss = float("inf")
        ckpt_dir = "./checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(
            ckpt_dir, f"best_flexi_swe_vae_{wandb.run.id}.pth"
        )

        for epoch in range(num_epochs):
            model.train()
            train_loss_rec = 0.0
            train_loss_dir = 0.0
            train_loss_kld = 0.0

            for batch in train_loader:
                x_t      = batch["x_t"].to(device)
                x_tau    = batch["x_t_tau"].to(device)
                tau_norm = batch["tau_norm"].to(device)

                optimizer.zero_grad()

                x_hat, x_hat_tau, mu, log_var, z, z_tau = model(x_t, tau_norm)

                loss_rec = criterion(x_hat, x_t)
                loss_dir = criterion(x_hat_tau, x_tau)
                loss_kld = beta_vae_kld(mu, log_var)

                loss = loss_rec + lambda_dir * loss_dir + beta * loss_kld
                loss.backward()
                optimizer.step()

                train_loss_rec += loss_rec.item()
                train_loss_dir += loss_dir.item()
                train_loss_kld += loss_kld.item()

            train_loss_rec /= len(train_loader)
            train_loss_dir /= len(train_loader)
            train_loss_kld /= len(train_loader)
            train_loss = (
                train_loss_rec
                + lambda_dir * train_loss_dir
                + beta * train_loss_kld
            )

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            model.eval()
            val_loss_rec = 0.0
            val_loss_dir = 0.0
            val_loss_kld = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    x_t      = batch["x_t"].to(device)
                    x_tau    = batch["x_t_tau"].to(device)
                    tau_norm = batch["tau_norm"].to(device)

                    x_hat, x_hat_tau, mu, log_var, z, z_tau = model(x_t, tau_norm)

                    loss_rec = criterion(x_hat, x_t)
                    loss_dir = criterion(x_hat_tau, x_tau)
                    loss_kld = beta_vae_kld(mu, log_var)

                    val_loss_rec += loss_rec.item()
                    val_loss_dir += loss_dir.item()
                    val_loss_kld += loss_kld.item()

            val_loss_rec /= len(val_loader)
            val_loss_dir /= len(val_loader)
            val_loss_kld /= len(val_loader)
            val_loss = (
                val_loss_rec
                + lambda_dir * val_loss_dir
                + beta * val_loss_kld
            )

            print(
                f"[Epoch {epoch+1}/{num_epochs}] "
                f"Train: total={train_loss:.4e}, rec={train_loss_rec:.4e}, "
                f"dir={train_loss_dir:.4e}, kld={train_loss_kld:.4e} | "
                f"Val: total={val_loss:.4e}, rec={val_loss_rec:.4e}, "
                f"dir={val_loss_dir:.4e}, kld={val_loss_kld:.4e} | "
                f"lr={current_lr:.2e}"
            )

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "lr": current_lr,
                    "train/total": train_loss,
                    "train/rec": train_loss_rec,
                    "train/dir": train_loss_dir,
                    "train/kld": train_loss_kld,
                    "val/total": val_loss,
                    "val/rec": val_loss_rec,
                    "val/dir": val_loss_dir,
                    "val/kld": val_loss_kld,
                },
                step=epoch + 1,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "u_mean": u_mean,
                        "u_std": u_std,
                        "beta": beta,
                        "lambda_dir": lambda_dir,
                        "latent_dim": cfg.latent_dim,
                    },
                    ckpt_path,
                )
                print(f"  -> New best val loss! Checkpoint saved to {ckpt_path}")
                wandb.summary["best_val_loss"] = best_val_loss

            if (epoch + 1) % 4 == 0:
                batch_vis = next(iter(val_loader))
                x_t_vis      = batch_vis["x_t"].to(device)
                x_tau_vis    = batch_vis["x_t_tau"].to(device)
                tau_norm_vis = batch_vis["tau_norm"].to(device)

                tau_int = batch_vis["tau"]
                t0_int  = batch_vis["t0"]

                with torch.no_grad():
                    x_hat_vis, x_hat_tau_vis, mu_vis, log_var_vis, z_vis, z_tau_vis = model(
                        x_t_vis, tau_norm_vis
                    )

                x_t_0       = x_t_vis[0, 0].cpu().numpy()
                x_tau_0     = x_tau_vis[0, 0].cpu().numpy()
                x_hat_0     = x_hat_vis[0, 0].cpu().numpy()
                x_hat_tau_0 = x_hat_tau_vis[0, 0].cpu().numpy()

                t0_0   = int(t0_int[0].item())
                tau_0  = int(tau_int[0].item())
                t_tp_0 = t0_0 + tau_0

                x_t_0       = x_t_0 * u_std + u_mean
                x_tau_0     = x_tau_0 * u_std + u_mean
                x_hat_0     = x_hat_0 * u_std + u_mean
                x_hat_tau_0 = x_hat_tau_0 * u_std + u_mean

                vmin = min(
                    x_t_0.min(), x_tau_0.min(), x_hat_0.min(), x_hat_tau_0.min()
                )
                vmax = max(
                    x_t_0.max(), x_tau_0.max(), x_hat_0.max(), x_hat_tau_0.max()
                )

                fig, axs = plt.subplots(2, 2, figsize=(10, 6))

                im0 = axs[0, 0].imshow(x_t_0, cmap="RdBu_r", vmin=vmin, vmax=vmax)
                axs[0, 0].set_title(f"GT: u(t), t = {t0_0}")
                fig.colorbar(im0, ax=axs[0, 0], shrink=0.8)

                im1 = axs[0, 1].imshow(x_tau_0, cmap="RdBu_r", vmin=vmin, vmax=vmax)
                axs[0, 1].set_title(f"GT: u(t+τ), t = {t_tp_0}, τ = {tau_0}")
                fig.colorbar(im1, ax=axs[0, 1], shrink=0.8)

                im2 = axs[1, 0].imshow(x_hat_0, cmap="RdBu_r", vmin=vmin, vmax=vmax)
                axs[1, 0].set_title("Recon: û(t)")
                fig.colorbar(im2, ax=axs[1, 0], shrink=0.8)

                im3 = axs[1, 1].imshow(
                    x_hat_tau_0, cmap="RdBu_r", vmin=vmin, vmax=vmax
                )
                axs[1, 1].set_title("Pred: û(t+τ)")
                fig.colorbar(im3, ax=axs[1, 1], shrink=0.8)

                plt.suptitle(
                    f"Epoch {epoch+1}: val viz (t0={t0_0}, τ={tau_0})", y=1.02
                )
                plt.tight_layout()

                wandb.log(
                    {"val/viz": wandb.Image(fig)},
                    step=epoch + 1,
                )
                plt.close(fig)


if __name__ == "__main__":
    train()
