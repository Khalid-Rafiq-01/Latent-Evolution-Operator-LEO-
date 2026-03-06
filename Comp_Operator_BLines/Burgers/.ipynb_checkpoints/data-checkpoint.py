import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataclasses import dataclass, asdict
import json

num_time_steps = 500
def get_dt(num_time_steps):
    return 2.0/num_time_steps

dt = get_dt(num_time_steps)


def exact_solution(Rnum,t):
    x = np.linspace(0.0,1.0,num=128)
    t0 = np.exp(Rnum/8.0)
    return (x/(t+1))/(1.0+np.sqrt((t+1)/t0)*np.exp(Rnum*(x*x)/(4.0*t+4)))


class ReDataset:
    def __init__(self, 
                X: np.ndarray = None,
                X_tau: np.ndarray = None,
                t_values: np.ndarray = None,
                tau_values: np.ndarray = None,
                Re_values: np.ndarray = None):
        self.X = X
        self.X_tau = X_tau
        self.t_values = t_values
        self.tau_values = tau_values
        self.Re_values = Re_values

    def append(self, other):
        self.X = np.concatenate([self.X, other.X]) if self.X is not None else other.X
        self.X_tau = np.concatenate([self.X_tau, other.X_tau]) if self.X_tau is not None else other.X_tau
        self.t_values = np.concatenate([self.t_values, other.t_values]) if self.t_values is not None else other.t_values
        self.tau_values = np.concatenate([self.tau_values, other.tau_values]) if self.tau_values is not None else other.tau_values
        self.Re_values = np.concatenate([self.Re_values, other.Re_values]) if self.Re_values is not None else other.Re_values

@dataclass
class IntervalSplit:
    interpolation: tuple
    extrapolation_left: tuple
    extrapolation_right: tuple


def prepare_Re_dataset(Re_range=(400, 2400), tau_range=(0, 500), dt=dt, n_samples=5000):
    X = []
    X_tau = []
    t_values = []
    tau_values = []
    Re_values = []
    TRANGE = (0,2) # Need to fix the Init Condition:
    while len(X) < n_samples:
        # sample Re log uniformly
        logRe = np.random.uniform(np.log(Re_range[0]), np.log(Re_range[1]))
        Re = np.exp(logRe).round().astype(int)
        t = 0.0 #np.random.uniform(*TRANGE)
        x_t = exact_solution(Re, t)
        # print('tau_range', tau_range)
        tau = np.random.randint(*tau_range) # any random value between 0-500 as t_0 = 0, => t_max = 2.0
        x_tau = exact_solution(Re, t+(tau*dt))
        
        X.append(x_t)
        X_tau.append(x_tau)
        t_values.append(t)
        tau_values.append(tau)
        Re_values.append(Re)

    X = np.array(X)
    X_tau = np.array(X_tau)
    t_values = np.array(t_values)
    tau_values = np.array(tau_values)
    Re_values = np.array(Re_values)
    # return X, X_tau, tau_values, Re_values
    dataset = ReDataset(X, X_tau, t_values, tau_values, Re_values)
    return dataset

def train_test_split_range(interval, interpolation_span=0.1, extrapolation_left_span=0.1, extrapolation_right_span=0.1):
    """
    Split the range into train and test ranges
    We have three test folds:
    1. Interpolation fold: Re and tau values are within the training (min, max) range but not in the training set
        We sample an interval of length x_interpolation_span% randomly from the total range
    2. Extrapolation fold: Re and tau values are outside the training (min, max) range
        We sample two intervals of length x_extrapolation_right_span% and x_extrapolation_left_span% from the total range
    3. Validation fold: Re and tau values are randomly sampled from the total set

    Overall interval looks like:
    Extrapolation_left_test | normal | Interpolation_test | normal | Extrapolation_right_test
    (min, extrapolation_left) | (extraplation_left, interpolation_min) | (interpolation_min, interpolation_max) | (interpolation_max, extrapolation_right) | (extrapolation_right, max)
    and
    train, val = split(normal, val_split)
    """
    r_min, r_max = interval
    length = (r_max-r_min)
    extra_left_length = extrapolation_left_span * length
    extra_right_length = extrapolation_right_span * length
    inter_length = interpolation_span * length

    extrapolation_left = (r_min, r_min + extra_left_length)
    extrapolation_right = (r_max - extra_right_length, r_max)

    interpolation_min = np.random.uniform(extrapolation_left[1], extrapolation_right[0] - inter_length)
    interpolation = (interpolation_min, interpolation_min + inter_length)

    train_ranges = [(extrapolation_left[1], interpolation[0]), (interpolation[1], extrapolation_right[0])]
    return IntervalSplit(interpolation, extrapolation_left, extrapolation_right), train_ranges

def get_train_ranges(interval_split):
    return [
        (interval_split.extrapolation_left[1], interval_split.interpolation[0]),
        (interval_split.interpolation[1], interval_split.extrapolation_right[0])
    ]

def get_train_val_test_folds(Re_range, 
                             re_interpolation_span=0.10,
                             re_extrapolation_left_span=0.10,
                             re_extrapolation_right_span=0.10,
                             n_samples_train=40_000,  # 2 ranges × 40,000 = 80,000 total
                             n_samples_val=8_000):     # 3 ranges × 8,000 = 24,000 total
    """
    Generate train and val datasets with NO TAU SPLITS.
    Only split on Re, use full tau range (0-500) for all samples.
    
    Returns:
        dataset_train  : ReDataset
        dataset_val    : ReDataset  
        Re_interval_split: IntervalSplit
    """

    # ---------------------------------------------------------------------
    # 1) Split Re into regions: left extrp, interp, right extrp, train
    # ---------------------------------------------------------------------
    Re_interval_split, Re_train_ranges = train_test_split_range(
        Re_range,
        re_interpolation_span,
        re_extrapolation_left_span,
        re_extrapolation_right_span
    )

    # -------------------------------------------------------------
    # 2) Build the TRAIN dataset from Re training ranges
    #    Use FULL TAU RANGE (0-500) for all training samples
    # -------------------------------------------------------------
    dataset_train = ReDataset()
    for re_subrange in Re_train_ranges:  # 2 intervals
        subset = prepare_Re_dataset(
            Re_range=re_subrange,
            tau_range=(0, 500),  # FULL TAU RANGE - no splits
            n_samples=n_samples_train
        )
        dataset_train.append(subset)

    # -------------------------------------------------------------
    # 3) Build the VAL dataset from Re validation intervals
    #    Use FULL TAU RANGE (0-500) for all validation samples
    # -------------------------------------------------------------
    re_val_intervals = [
        Re_interval_split.extrapolation_left,
        Re_interval_split.interpolation,
        Re_interval_split.extrapolation_right
    ]

    dataset_val = ReDataset()
    for re_val_range in re_val_intervals:
        subset_val = prepare_Re_dataset(
            Re_range=re_val_range,
            tau_range=(0, 500),  # FULL TAU RANGE - no splits
            n_samples=n_samples_val
        )
        dataset_val.append(subset_val)

    # Return None for tau_interval_split since we're not splitting tau
    return dataset_train, dataset_val, Re_interval_split, None

def plot_sample(dataset, i):
    X = dataset.X
    X_tau = dataset.X_tau
    Tau = dataset.tau_values
    Re_total = dataset.Re_values
    plt.plot(X[i], label = "Initial State")
    plt.plot(X_tau[i], label = "Mapped State")
    plt.title(f'Tau: {Tau[i]}, Re: {Re_total[i]}')
    plt.legend()
    plt.show()

def save_to_path(path, dataset_train, dataset_val, Re_interval_split, tau_interval_split):
    if not os.path.exists(path):
        os.makedirs(path)
    # save dataset_train, dataset_val, Re_interval_split, tau_interval_split to pkl files
    dataset_train_path = os.path.join(path, 'dataset_train.pkl')
    dataset_val_path = os.path.join(path, 'dataset_val.pkl')
    Re_interval_split_path = os.path.join(path, 'Re_interval_split.json')
    # We don't save tau_interval_split since we're not using it

    with open(dataset_train_path, 'wb') as f:
        pickle.dump(dataset_train, f)
    with open(dataset_val_path, 'wb') as f:
        pickle.dump(dataset_val, f)

    with open(Re_interval_split_path, 'w') as f:
        json.dump(asdict(Re_interval_split), f)

def load_from_path(path):
    dataset_train_path = os.path.join(path, 'dataset_train.pkl')
    dataset_val_path = os.path.join(path, 'dataset_val.pkl')
    Re_interval_split_path = os.path.join(path, 'Re_interval_split.json')

    with open(dataset_train_path, 'rb') as f:
        dataset_train = pickle.load(f)
    with open(dataset_val_path, 'rb') as f:
        dataset_val = pickle.load(f)
    with open(Re_interval_split_path, 'r') as f:
        Re_interval_split = json.load(f)
        Re_interval_split = IntervalSplit(**Re_interval_split)

    return dataset_train, dataset_val, Re_interval_split, None

def main():
    Re_range = (400, 2400)
    num_time_steps = 500
    dataset_train, dataset_val, Re_interval_split, _ = get_train_val_test_folds(Re_range)
    
    print(f"Training data size: {len(dataset_train.X)}")
    print(f"Validation data size: {len(dataset_val.X)}")
    print(f"Re Interval Split: {Re_interval_split}")
    
    # Verify tau ranges
    print(f"Training tau range: {dataset_train.tau_values.min()} to {dataset_train.tau_values.max()}")
    print(f"Validation tau range: {dataset_val.tau_values.min()} to {dataset_val.tau_values.max()}")
    
    save_to_path('data', dataset_train, dataset_val, Re_interval_split, None)

if __name__ == '__main__':
    main()