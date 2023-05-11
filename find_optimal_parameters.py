import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import torch
import math
import matplotlib.pyplot as plt

import miscellaneous.datasets as datasets
import miscellaneous.utilities as utilities
import miscellaneous.metrics as metrics
from variational import operators, solvers

import astra

######################## PARSING ###########################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data',
                    help="Choose the dataset.",
                    type=str,
                    required=False,
                    default="COULE",
                    choices=["COULE"]
                    )
parser.add_argument('--p',
                    help="Set the sparsity parameter p.",
                    type=float,
                    required=False,
                    default=1
                    )
parser.add_argument('--experiment', '-e',
                    help="Set the experiment angular sparsity.",
                    type=str,
                    required=False,
                    default="mild",
                    choices=["mild", "severe"])
parser.add_argument('--mode',
                    help="Choose if run the algorithm on training or test set.",
                    type=str,
                    required=False,
                    default="train",
                    choices=["train", "test"]
                    )
parser.add_argument('--idx',
                    help="Id of the image to search the parameters.",
                    type=int,
                    required=False,
                    default=0)

args = parser.parse_args()

######################## INITIALIZATION ###########################
# Get the path for the data
path = f'./data/{args.data}/{args.mode}/gt/'

# Get configuration path.
cfg = utilities.load_toml(args.data)

if args.experiment == "mild":
    test_cfg = cfg['test_mild_sparsity']
elif args.experiment == "severe":
    test_cfg = cfg['test_severe_sparsity']
else:
    raise ValueError

# Define dataset 
data = datasets.ImageDataset(path, numpy=True)

# Get data shape.
m, n = cfg['m'], cfg['n']

# Define the operator A
angles = np.linspace(np.deg2rad(test_cfg['start_angle']), np.deg2rad(test_cfg['end_angle']), 
                     test_cfg['n_angles'], endpoint=False)

A = operators.CTProjector((m, n), angles, det_size=test_cfg['det_size'], geometry=test_cfg['geometry'])

######################## GRID-SEARCH ###########################
# Get true data from the dataset.
x_true = data[args.idx] # Shape: (1, m, n)

# Compute the sinogram
y = A(x_true.flatten())

# Add noise
y_delta = y + utilities.get_gaussian_noise(y, noise_level=test_cfg['noise_level'])

# Setup the search space
lmbda_space = np.linspace(8e-4, 2e-3, 20)
epsilon_scale_space = [5e-3] # [1e-3, 5e-3, 1e-4, 5e-4, 1e-5]

RMSE_vec = np.zeros((len(lmbda_space), len(epsilon_scale_space)))
for i, lmbda in enumerate(lmbda_space):
    for j, epsilon_scale in enumerate(epsilon_scale_space):
        # Define epsilon
        epsilon = epsilon_scale * np.max(y_delta) * np.sqrt(len(y_delta))

        # Initialize the solver
        solver = solvers.ChambollePockTpV(A)

        # Compute solution
        x_tpv = solver(y_delta, epsilon=epsilon, lmbda=lmbda, x_true=x_true.flatten(), p=args.p, maxiter=400).reshape((m, n))
        
        # Compute (and save) RMSE
        RMSE_vec[i, j] = metrics.RMSE(x_tpv, x_true[0])

        # Verbose
        print(f"({lmbda}, {epsilon_scale}) RMSE: {RMSE_vec[i, j]}.")

# Save the result
out_path = f'./results/optimality/{args.data}/{args.mode}'
utilities.create_path_if_not_exists(out_path)

np.save(out_path + f"/RMSE_optimal_{args.experiment}_p_{args.p}.npy", RMSE_vec)