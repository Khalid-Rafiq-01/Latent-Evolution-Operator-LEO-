from dataclasses import dataclass
import json

@dataclass
class Config:
    # default values. DO NOT TOUCH
    name: str = 'FVAE_IC_Fixed'
    latent_dim: int = 2
    input_dim: int = 128
    batch_size: int = 128
    lr: float = 3e-4
    num_epochs: int = 200
    #n_samples_train: int = 80_000  # Updated to match your actual data size

    # REMOVED: tau split parameters since we don't use them anymore
    # num_time_steps: int = 500
    # tau_left_fraction: float = 0.35
    # tau_right_fraction: float = 0.85

    gamma: float = 3.25
    beta: float = 1e-4

    val_every: float = 0.25
    plot_train_every: float = 0.01

    save_dir: str = 'checkpoints'

def load_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return Config(**config)