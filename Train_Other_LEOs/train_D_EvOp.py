import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import wandb

from data import load_from_path, ReDataset, exact_solution
from ev_op_models import Encoder, Decoder, Propagator  # AE-style backbone

warnings.filterwarnings("ignore")


# Utility: deterministic-ish
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# Data loader helper
def get_data_loader(dataset, batch_size, shuffle=True):
    data = list(zip(dataset.X,
                    dataset.X_tau,
                    dataset.t_values,
                    dataset.tau_values,
                    dataset.Re_values))
    # drop last incomplete batch
    data = data[: len(data) - len(data) % batch_size]
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True,
    )

# Plot a few validation examples to wandb
def log_val_examples(model, propagator, val_loader, device, epoch, num_examples=3):
    model.eval()
    propagator.eval()

    # materialize one pass of val batches
    val_batches = list(val_loader)
    if len(val_batches) == 0:
        return

    for eg_id in range(num_examples):
        # pick a random batch and index
        X, X_tau, t_vals, tau_vals, Re_vals = random.choice(val_batches)
        i = np.random.randint(0, X.shape[0])

        X_b = torch.as_tensor(X[i:i+1], dtype=torch.float32, device=device)
        Xtau_b = torch.as_tensor(X_tau[i:i+1], dtype=torch.float32, device=device)
        tau_b = torch.as_tensor(tau_vals[i:i+1], dtype=torch.float32, device=device)
        Re_b = torch.as_tensor(Re_vals[i:i+1], dtype=torch.float32, device=device)

        with torch.no_grad():
            # encoder-decoder at t
            x_hat_t, z_t = model(X_b)  # AE: returns (recon, z)
            # latent propagation
            z_prop = propagator(
                z_t,
                tau_b.view(-1, 1),
                Re_b.view(-1, 1)
            )
            x_hat_tau_prop = model.decoder(z_prop)

        x = X_b.squeeze().cpu().numpy()
        x_tau = Xtau_b.squeeze().cpu().numpy()
        xh_t = x_hat_t.squeeze().cpu().numpy()
        xh_tau_prop = x_hat_tau_prop.squeeze().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(x, label="GT: u(t)", lw=2, alpha=0.6)
        ax.plot(x_tau, label="GT: u(t+τ)", lw=2, alpha=0.6)
        ax.plot(xh_t, label="Recon: D(E(u(t)))", lw=1.5)
        ax.plot(xh_tau_prop, label="Pred: D(P(E(u(t)),τ,Re))", lw=1.8)

        ax.set_title(
            f"Epoch {epoch} | τ={float(tau_b.item()):.3f} | Re={float(Re_b.item()):.0f}",
            fontsize=10
        )
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        wandb.log({f"val/example_{eg_id+1}": wandb.Image(fig)}, step=epoch)
        plt.close(fig)


# Main training func (for sweeps)
def train():
    set_seed(42)

    run = wandb.init(project="AE_Direct_EvOp", config={
        "batch_size": 128,
        "lr": 3e-4,
        "epochs": 150,
        "zeta": 1.0,
        "clip_norm": 5.0,
    })
    config = wandb.config

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data 
    data_path = "./data"
    dataset_train, dataset_val, re_interval_split, tau_interval_split = load_from_path(data_path)

    train_loader = get_data_loader(dataset_train, batch_size=config.batch_size, shuffle=True)
    val_loader = get_data_loader(dataset_val,   batch_size=config.batch_size, shuffle=False)

    # model: AE backbone + propagator
    encoder = Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    # model should behave like: x_hat, z = model(X)
    class AEWrapper(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.encoder = enc
            self.decoder = dec

        def forward(self, x):
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return x_hat, z

    model = AEWrapper(encoder, decoder).to(DEVICE)
    propagator = Propagator().to(DEVICE)

    # optim + scheduler
    params = list(model.parameters()) + list(propagator.parameters())
    optim = torch.optim.Adam(params, lr=config.lr)
    # Cosine LR decay over total epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=config.epochs
    )

    mse = nn.MSELoss()

    # training loop 
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        propagator.train()

        tr_RL1 = 0.0
        tr_DPC = 0.0
        tr_ALIGN = 0.0  # just to monitor latent mismatch
        tr_LOSS = 0.0
        n_tr = 0

        for X, X_tau, t_train, tau_train, Re_train in train_loader:
            X = torch.as_tensor(X,       dtype=torch.float32, device=DEVICE)
            X_tau = torch.as_tensor(X_tau, dtype=torch.float32, device=DEVICE)
            tau = torch.as_tensor(tau_train, dtype=torch.float32, device=DEVICE)  # (B,)
            Re = torch.as_tensor(Re_train,  dtype=torch.float32, device=DEVICE)  # (B,)

            optim.zero_grad(set_to_none=True)

            # AE reconstruction at t and t+τ
            x_hat_t, z_t = model(X)        # (B, n), (B, m)
            x_hat_tau, z_tau = model(X_tau)  # not used in loss, but used to monitor alignment

            # latent propagation and decoded prediction
            z_prop = propagator(
                z_t,
                tau.view(-1, 1),
                Re.view(-1, 1)
            )
            u_hat_tau = model.decoder(z_prop)

            # losses (Direct EvOp / EvOp-Base)
            RL1 = mse(x_hat_t, X)
            ALIGN = mse(z_prop, z_tau)  # tracked only; not enforced
            DPC = mse(u_hat_tau, X_tau)

            loss = RL1 + config.zeta * DPC
            loss.backward()

            if config.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, config.clip_norm)

            optim.step()

            tr_RL1 += RL1.item()
            tr_DPC += DPC.item()
            tr_ALIGN += ALIGN.item()
            tr_LOSS += loss.item()
            n_tr += 1

        # normalize train metrics
        tr_RL1 /= max(1, n_tr)
        tr_DPC /= max(1, n_tr)
        tr_ALIGN /= max(1, n_tr)
        tr_LOSS /= max(1, n_tr)

        # validation 
        model.eval()
        propagator.eval()

        va_RL1 = va_DPC = va_ALIGN = va_LOSS = 0.0
        n_va = 0

        with torch.no_grad():
            for X, X_tau, t_val, tau_val, Re_val in val_loader:
                X = torch.as_tensor(X,       dtype=torch.float32, device=DEVICE)
                X_tau = torch.as_tensor(X_tau, dtype=torch.float32, device=DEVICE)
                tau = torch.as_tensor(tau_val, dtype=torch.float32, device=DEVICE)
                Re = torch.as_tensor(Re_val,  dtype=torch.float32, device=DEVICE)

                x_hat_t, z_t = model(X)
                x_hat_tau, z_tau = model(X_tau)

                z_prop = propagator(
                    z_t,
                    tau.view(-1, 1),
                    Re.view(-1, 1)
                )
                u_hat_tau = model.decoder(z_prop)

                RL1 = mse(x_hat_t, X)
                ALIGN = mse(z_prop, z_tau)
                DPC = mse(u_hat_tau, X_tau)

                vloss = RL1 + config.zeta * DPC

                va_RL1 += RL1.item()
                va_DPC += DPC.item()
                va_ALIGN += ALIGN.item()
                va_LOSS += vloss.item()
                n_va += 1

        va_RL1 /= max(1, n_va)
        va_DPC /= max(1, n_va)
        va_ALIGN /= max(1, n_va)
        va_LOSS /= max(1, n_va)

        # step LR scheduler
        scheduler.step()

        # log to wandb 
        wandb.log({
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train/loss": tr_LOSS,
            "train/RL1": tr_RL1,
            "train/DPC": tr_DPC,
            "train/ALIGN": tr_ALIGN,
            "val/loss": va_LOSS,
            "val/RL1": va_RL1,
            "val/DPC": va_DPC,
            "val/ALIGN": va_ALIGN,
        }, step=epoch)

        # also log some qualitative plots each epoch
        log_val_examples(model, propagator, val_loader, DEVICE, epoch, num_examples=3)

        # track best
        if va_LOSS < best_val_loss:
            best_val_loss = va_LOSS
            # optional: save best weights per run
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "propagator": propagator.state_dict(),
            }, os.path.join(wandb.run.dir, "best_model.pt"))

        # small console print
        if (epoch % 20 == 0) or (epoch == 1):
            print(
                f"[Epoch {epoch:03d}] "
                f"train: Loss={tr_LOSS:.3e} | RL1={tr_RL1:.3e} | DPC={tr_DPC:.3e} | ALIGN={tr_ALIGN:.3e} || "
                f"val: Loss={va_LOSS:.3e} | RL1={va_RL1:.3e} | DPC={va_DPC:.3e} | ALIGN={va_ALIGN:.3e}"
            )

    run.finish()


if __name__ == "__main__":
    train()
