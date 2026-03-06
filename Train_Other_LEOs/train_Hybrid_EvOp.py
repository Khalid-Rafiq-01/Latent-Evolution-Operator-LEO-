import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import wandb

from data import load_from_path, ReDataset, exact_solution
from ev_op_models import Encoder, Decoder, Propagator, VAE  # VAE used as AE-style backbone

warnings.filterwarnings("ignore")


# Utils
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_data_loader(dataset, batch_size, shuffle=True):
    data = list(zip(
        dataset.X,
        dataset.X_tau,
        dataset.t_values,
        dataset.tau_values,
        dataset.Re_values,
    ))
    # drop incomplete batch
    data = data[: len(data) - len(data) % batch_size]
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True,
    )


def lambda_align_weight(epoch, total_epochs, lambda_align_max, warmup_frac: float):
    """
    Linear ramp for latent-consistency weight.

    - epoch in [1, total_epochs]
    - warmup_frac in [0,1): fraction of epochs with λ_align = 0
    """
    e = float(epoch)
    T = float(total_epochs)
    warmup_epochs = warmup_frac * T

    if e <= warmup_epochs:
        return 0.0

    # normalized progress after warmup
    denom = max(1.0, T - warmup_epochs)
    frac = (e - warmup_epochs) / denom  # in [0,1]
    w = frac * float(lambda_align_max)
    return float(min(lambda_align_max, max(0.0, w)))


def log_val_examples(model, propagator, val_loader, device, epoch,
                     lambda_align, lambda_direct, num_examples=3):
    """
    Log a few qualitative val plots to wandb.

    Shows:
      - GT u(t), u(t+τ)
      - Recon D(E(u(t))), D(E(u(t+τ)))
      - Pred D(P(E(u(t)), τ, Re))
    """
    model.eval()
    propagator.eval()

    val_batches = list(val_loader)
    if not val_batches:
        return

    for eg_id in range(num_examples):
        X, X_tau, t_vals, tau_vals, Re_vals = random.choice(val_batches)
        i = np.random.randint(0, X.shape[0])

        X_b = torch.as_tensor(X[i:i+1], dtype=torch.float32, device=device)
        Xtau_b = torch.as_tensor(X_tau[i:i+1], dtype=torch.float32, device=device)
        tau_b = torch.as_tensor(tau_vals[i:i+1], dtype=torch.float32, device=device)
        Re_b = torch.as_tensor(Re_vals[i:i+1], dtype=torch.float32, device=device)

        with torch.no_grad():
            x_hat_t, z_t = model(X_b)
            x_hat_tau, z_tau = model(Xtau_b)

            z_prop = propagator(
                z_t,
                tau_b.view(-1, 1),
                Re_b.view(-1, 1),
            )
            x_hat_tau_prop = model.decoder(z_prop)

        x = X_b.squeeze().cpu().numpy()
        x_tau = Xtau_b.squeeze().cpu().numpy()
        xh_t = x_hat_t.squeeze().cpu().numpy()
        xh_tau = x_hat_tau.squeeze().cpu().numpy()
        xh_tau_prop = x_hat_tau_prop.squeeze().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(x, label="GT: u(t)", lw=2, alpha=0.6)
        ax.plot(x_tau, label="GT: u(t+τ)", lw=2, alpha=0.6)
        ax.plot(xh_t, label="Recon: D(E(u(t)))", lw=1.5)
        ax.plot(xh_tau, label="Recon: D(E(u(t+τ)))", lw=1.5, alpha=0.9)
        ax.plot(xh_tau_prop,
                label="Pred: D(P(E(u(t)),τ,Re))",
                lw=1.8)

        ax.set_title(
            f"Epoch {epoch} | λ_align={lambda_align:.3f}, λ_direct={lambda_direct:.3f} | "
            f"τ={float(tau_b.item()):.3f} | Re={float(Re_b.item()):.0f}",
            fontsize=9,
        )
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        wandb.log({f"val/example_{eg_id+1}": wandb.Image(fig)}, step=epoch)
        plt.close(fig)


# Main training (Hybrid EvOp)

def train():
    set_seed(42)

    run = wandb.init(
        project="AE_Hybrid_EvOp",
        config=dict(
            batch_size=128,
            lr=3e-4,
            epochs=150,
            lambda_align_max=1.0,
            warmup_frac=0.4,      # 40% AE-only; λ_align = 0
            lambda_direct=1.0,    # weight on direct D(P(z))-vs-u(t+τ)
            clip_norm=5.0,
        ),
    )
    config = wandb.config

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data 
    data_path = "./data"
    dataset_train, dataset_val, re_split, tau_split = load_from_path(data_path)

    train_loader = get_data_loader(dataset_train,
                                   batch_size=config.batch_size,
                                   shuffle=True)
    val_loader = get_data_loader(dataset_val,
                                 batch_size=config.batch_size,
                                 shuffle=False)

    # Models 
    encoder = Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)

    # VAE wrapper: forward(x) -> (x_hat, z)
    model = VAE(encoder, decoder).to(DEVICE)
    propagator = Propagator().to(DEVICE)

    params = list(model.parameters()) + list(propagator.parameters())
    optim = torch.optim.Adam(params, lr=config.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=config.epochs,
    )

    mse = nn.MSELoss()

    # best model AFTER λ_align is active (true hybrid)
    best_val_loss_hybrid = float("inf")
    best_epoch_hybrid = None

    # Training loop
    for epoch in range(1, config.epochs + 1):
        model.train()
        propagator.train()

        lam_align = lambda_align_weight(
            epoch,
            total_epochs=config.epochs,
            lambda_align_max=config.lambda_align_max,
            warmup_frac=config.warmup_frac,
        )
        lam_direct = float(config.lambda_direct)

        tr_RL1 = tr_RL2 = tr_ALIGN = tr_DCP = tr_LOSS = 0.0
        n_tr = 0

        for X, X_tau, t_train, tau_train, Re_train in train_loader:
            X = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
            X_tau = torch.as_tensor(X_tau, dtype=torch.float32, device=DEVICE)
            tau = torch.as_tensor(tau_train, dtype=torch.float32, device=DEVICE)
            Re = torch.as_tensor(Re_train, dtype=torch.float32, device=DEVICE)

            optim.zero_grad(set_to_none=True)

            # encode/decode both times
            x_hat_t, z_t = model(X)
            x_hat_tau, z_tau = model(X_tau)

            # propagate & decode
            z_prop = propagator(
                z_t,
                tau.view(-1, 1),
                Re.view(-1, 1),
            )
            u_hat_tau = model.decoder(z_prop)

            # losses
            RL1 = mse(x_hat_t, X)
            RL2 = mse(x_hat_tau, X_tau)
            ALIGN_L = mse(z_prop, z_tau.detach())
            DCP_L = mse(u_hat_tau, X_tau)

            loss = RL1 + RL2 + lam_align * ALIGN_L + lam_direct * DCP_L
            loss.backward()

            if config.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, config.clip_norm)

            optim.step()

            tr_RL1 += RL1.item()
            tr_RL2 += RL2.item()
            tr_ALIGN += ALIGN_L.item()
            tr_DCP += DCP_L.item()
            tr_LOSS += loss.item()
            n_tr += 1

        tr_RL1 /= max(1, n_tr)
        tr_RL2 /= max(1, n_tr)
        tr_ALIGN /= max(1, n_tr)
        tr_DCP /= max(1, n_tr)
        tr_LOSS /= max(1, n_tr)

        # Validation
        model.eval()
        propagator.eval()

        va_RL1 = va_RL2 = va_ALIGN = va_DCP = va_LOSS = 0.0
        n_va = 0

        with torch.no_grad():
            for X, X_tau, t_val, tau_val, Re_val in val_loader:
                X = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
                X_tau = torch.as_tensor(X_tau, dtype=torch.float32, device=DEVICE)
                tau = torch.as_tensor(tau_val, dtype=torch.float32, device=DEVICE)
                Re = torch.as_tensor(Re_val, dtype=torch.float32, device=DEVICE)

                x_hat_t, z_t = model(X)
                x_hat_tau, z_tau = model(X_tau)

                z_prop = propagator(
                    z_t,
                    tau.view(-1, 1),
                    Re.view(-1, 1),
                )
                u_hat_tau = model.decoder(z_prop)

                RL1 = mse(x_hat_t, X)
                RL2 = mse(x_hat_tau, X_tau)
                ALIGN_L = mse(z_prop, z_tau)
                DCP_L = mse(u_hat_tau, X_tau)

                vloss = RL1 + RL2 + lam_align * ALIGN_L + lam_direct * DCP_L

                va_RL1 += RL1.item()
                va_RL2 += RL2.item()
                va_ALIGN += ALIGN_L.item()
                va_DCP += DCP_L.item()
                va_LOSS += vloss.item()
                n_va += 1

        va_RL1 /= max(1, n_va)
        va_RL2 /= max(1, n_va)
        va_ALIGN /= max(1, n_va)
        va_DCP /= max(1, n_va)
        va_LOSS /= max(1, n_va)

        scheduler.step()

        # Log to wandb
        wandb.log(
            {
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                "lambda_align": lam_align,
                "lambda_direct": lam_direct,
                # train
                "train/loss": tr_LOSS,
                "train/RL1": tr_RL1,
                "train/RL2": tr_RL2,
                "train/ALIGN": tr_ALIGN,
                "train/DCP": tr_DCP,
                # val
                "val/loss": va_LOSS,
                "val/RL1": va_RL1,
                "val/RL2": va_RL2,
                "val/ALIGN": va_ALIGN,
                "val/DCP": va_DCP,
            },
            step=epoch,
        )

        log_val_examples(
            model, propagator, val_loader, DEVICE, epoch,
            lambda_align=lam_align, lambda_direct=lam_direct, num_examples=3
        )

        # Checkpoint logic (true hybrid)
        # Save only once λ_align is active, so we don't pick pure-AE or pure-direct.
        if lam_align > 0.0:
            if va_LOSS < best_val_loss_hybrid:
                best_val_loss_hybrid = va_LOSS
                best_epoch_hybrid = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "propagator": propagator.state_dict(),
                        "lambda_align_max": float(config.lambda_align_max),
                        "lambda_direct": float(config.lambda_direct),
                        "warmup_frac": float(config.warmup_frac),
                        "best_val_loss_hybrid": float(best_val_loss_hybrid),
                    },
                    os.path.join(wandb.run.dir, "best_model_hybrid.pt"),
                )

        if (epoch % 20 == 0) or (epoch == 1):
            print(
                f"[Epoch {epoch:03d}] "
                f"λ_align={lam_align:.3f}, λ_direct={lam_direct:.3f} | "
                f"train: Loss={tr_LOSS:.3e} | RL1={tr_RL1:.3e} | RL2={tr_RL2:.3e} "
                f"| ALIGN={tr_ALIGN:.3e} | DCP={tr_DCP:.3e} || "
                f"val: Loss={va_LOSS:.3e} | RL1={va_RL1:.3e} | RL2={va_RL2:.3e} "
                f"| ALIGN={va_ALIGN:.3e} | DCP={va_DCP:.3e}"
            )

    if best_epoch_hybrid is not None:
        wandb.summary["best_val_loss_hybrid"] = best_val_loss_hybrid
        wandb.summary["best_epoch_hybrid"] = best_epoch_hybrid

    run.finish()


if __name__ == "__main__":
    train()
