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
args = parser.parse_args()

######################## INITIALIZATION ###########################
# Get the path for the data
path = f'./data/{args.data}/{args.mode}/gt/'
out_path = f'./data/{args.data}/{args.mode}/IS_{args.experiment}_p_{args.p}'

# Since we save the FBP solutions, generate the correspoinding results folder.
FBP_path = f'./results/{args.data}/{args.mode}/FBP_{args.experiment}'

# Create output folders if not exists
utilities.create_path_if_not_exists(out_path)
utilities.create_path_if_not_exists(FBP_path)

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

######################## OPTIMIZATION ###########################
name_list = sorted([f"{i}.png" for i in range(len(data))])

ssim_FBP_vec = np.zeros((len(data), ))
ssim_vec = np.zeros((len(data), ))
for i in range(len(data)):
    # Get true data from the dataset.
    x_true = data[i] # Shape: (1, m, n)

    # Compute the sinogram
    y = A(x_true.flatten())

    # Add noise
    y_delta = y + utilities.get_gaussian_noise(y, noise_level=test_cfg['noise_level'])

    # Get FBP reconstruction (for comparison).
    x_fbp = A.FBP(y_delta.flatten()).reshape((m, n))
    plt.imsave(f'{FBP_path}/{name_list[i]}', x_fbp, cmap='gray')


    # Get the parameters for the optimization algorithm
    epsilon = test_cfg[f'p_{str(args.p).replace(".", "_")}']['epsilon_scale'] * np.max(y_delta) * np.sqrt(len(y_delta))
    lmbda = test_cfg[f'p_{str(args.p).replace(".", "_")}']['lmbda']

    # Initialize the solver
    solver = solvers.ChambollePockTpV(A)

    # Compute solution
    x_tpv = solver(y_delta, epsilon=epsilon, lmbda=lmbda, x_true=x_true.flatten(), p=args.p, maxiter=200).reshape((m, n))

    # Save the result
    plt.imsave(f'{out_path}/{name_list[i]}', x_tpv.reshape((m, n)), cmap='gray')
    
    # Compute (and save) metrics
    ssim_FBP_vec[i] = metrics.SSIM(x_fbp, x_true[0])
    ssim_vec[i] = metrics.SSIM(x_tpv, x_true[0])

    # Verbose
    print(f"{i+1}/{len(data)}. FBP SSIM: {ssim_FBP_vec[i]}, TpV SSIM: {ssim_vec[i]}.")

# Save the metrics
ssim_path = f'./results/metrics/{args.data}/{args.mode}'

utilities.create_path_if_not_exists(ssim_path)

np.save(f"{ssim_path}/SSIM_FBP_{args.experiment}.npy", ssim_FBP_vec)
np.save(f"{ssim_path}/SSIM_IS_{args.experiment}_p_{args.p}.npy", ssim_vec)