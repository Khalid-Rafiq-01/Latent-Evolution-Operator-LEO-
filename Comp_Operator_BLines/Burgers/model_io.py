import torch
from dataclasses import dataclass, asdict
from data import IntervalSplit
from config import Config


def save_model(path, model, tau_interval_split, re_interval_split, config):
    save_dict = {
        'model_state_dict': model.state_dict(),
        're_interval_split': asdict(re_interval_split),
        'config': asdict(config),
    }
    
    # Only add tau_interval_split if it's not None
    if tau_interval_split is not None:
        save_dict['tau_interval_split'] = asdict(tau_interval_split)
    
    torch.save(save_dict, path)


def load_model(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Handle the case where tau_interval_split might be missing (for backward compatibility)
    tau_interval_split = None
    if 'tau_interval_split' in checkpoint and checkpoint['tau_interval_split'] is not None:
        tau_interval_split = IntervalSplit(**checkpoint['tau_interval_split'])
    
    re_interval_split = IntervalSplit(**checkpoint['re_interval_split'])
    config = Config(**checkpoint['config'])
    return model, re_interval_split, tau_interval_split, config