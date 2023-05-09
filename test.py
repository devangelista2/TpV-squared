import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.functional as F

import miscellaneous.datasets as datasets
from torch.utils.data import DataLoader

from models.architectures import UNet, AttnUNet
import miscellaneous.utilities as utilities
from variational import operators, solvers

from miscellaneous import metrics

############################### Parsing ################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data',
                    required=False,
                    help="Name of the dataset. Default: COULE.",
                    choices=["COULE"],
                    default='COULE')
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
parser.add_argument('--model',
                    help="Choose the model architecture.",
                    type=str,
                    required=False,
                    default="unet",
                    choices=["unet", "attnunet"]
                    )
parser.add_argument('--device',
                    required=False,
                    help="The device where the code is executed.",
                    type=str,
                    default="cpu",
                    choices=["cpu", "cuda"])
args = parser.parse_args()


############################### Load informations ################
# Define the device
device = args.device

# Get configuration path.
cfg = utilities.load_toml(args.data)

if args.experiment == "mild":
    test_cfg = cfg['test_mild_sparsity']
elif args.experiment == "severe":
    test_cfg = cfg['test_severe_sparsity']
else:
    raise ValueError

# Define paths
gt_path = f'./data/{args.data}/{args.mode}/gt/'
is_path = f'./data/{args.data}/{args.mode}/IS_{args.experiment}_p_{args.p}/'

gt_data = datasets.ImageDataset(gt_path,
                                numpy=False
                                )
is_data = datasets.ImageDataset(is_path,
                                numpy=False
                                )

# Get data shape.
m, n = cfg['m'], cfg['n']

######################## Define forward problem parameters
# Define the operator A
angles = np.linspace(np.deg2rad(test_cfg['start_angle']), np.deg2rad(test_cfg['end_angle']), 
                     test_cfg['n_angles'], endpoint=False)

A = operators.CTProjector((m, n), angles, det_size=test_cfg['det_size'], geometry=test_cfg['geometry'])

# Initialize the solver
solver = solvers.ChambollePockTpV(A)

######################## Define train parameters
if args.model == "unet":
    model_suffix = "UNet"
    model = UNet(img_ch=1, output_ch=1).to(device)
elif args.model == "attnunet":
    model_suffix = "AttnUNet"
    model = AttnUNet(img_ch=1, output_ch=1).to(device)
else:
    raise ValueError

# Define save path
rising_save_path = f"./results/{args.data}/{args.mode}/RISING_{args.experiment}_p_{args.p}/"
rising_is_save_path = f"./results/{args.data}/{args.mode}/RISING_IS_{args.experiment}_p_{args.p}/"

utilities.create_path_if_not_exists(rising_save_path)
utilities.create_path_if_not_exists(rising_is_save_path)

# Get name list sorted alphabetically
name_list = sorted([f"{i}.png" for i in range(len(gt_data))])

# Get the weights
weights_path = f'./model_weights/{args.data}/RISING_{args.experiment}_p_{args.p}/'

# Load the weights of the model
model.load_state_dict(torch.load(weights_path+model_suffix+'.pt'))
model = model.to(device)

# From gt_data, generate ris_data
rising_data = torch.zeros((len(gt_data), 1, m, n))
rising_is_data = torch.zeros((len(gt_data), 1, m, n))
for i in range(len(gt_data)):
    x_tmp = gt_data[i].numpy().flatten()
    
    # Compute the sinogram
    y = A(x_tmp)

    # Add noise
    y_delta = y + utilities.get_gaussian_noise(y, noise_level=test_cfg['noise_level'])

    # Get the parameters for the optimization algorithm
    epsilon = test_cfg[f'p_{str(args.p).replace(".", "_")}']['epsilon_scale'] * np.max(y_delta) * np.sqrt(len(y_delta))
    lmbda = test_cfg[f'p_{str(args.p).replace(".", "_")}']['lmbda']

    # Compute solution
    x_ris = torch.tensor(solver(y_delta, epsilon=epsilon, lmbda=lmbda, 
                               x_true=x_tmp, p=args.p, 
                               maxiter=test_cfg['k']).reshape((1, m, n)), requires_grad=True)
    x_ris = x_ris.type(torch.FloatTensor)
    
    # Send in GPU
    x_ris = x_ris.to(device)

    # Generate reconstruction
    x_rising = model(x_ris.unsqueeze(0))

    # Copy results in CPU
    x_rising = x_rising.detach().cpu().numpy()
    x_rising = utilities.normalize(x_rising)

    # From x_rising, compute x_rising_is
    x_rising_is = solver(y_delta, epsilon=epsilon, lmbda=lmbda, 
                         x_true=x_tmp, p=args.p, starting_point=np.expand_dims(x_rising.flatten(), -1),
                         maxiter=50).reshape((1, m, n))
    
    # Save into an array
    # rising_data[i] = x_rising
    # rising_is_data[i] = x_rising_is

    # Save in png
    plt.imsave(rising_save_path + name_list[i], x_rising.reshape((m, n)), cmap='gray')
    plt.imsave(rising_is_save_path + name_list[i], x_rising_is.reshape((m, n)), cmap='gray')

    # Compute metrics
    ssim_INGGT = metrics.SSIM(x_rising.reshape((m, n)), x_tmp.reshape((m, n)))
    ssim_INGISGT = metrics.SSIM(x_rising_is.reshape((m, n)), x_tmp.reshape((m, n)))
    ssim_INGINGIS = metrics.SSIM(x_rising.reshape((m, n)), x_rising_is.reshape((m, n)))

    re_INGGT = metrics.np_RE(x_rising.reshape((m, n)), x_tmp.reshape((m, n)))
    re_INGISGT = metrics.np_RE(x_rising_is.reshape((m, n)), x_tmp.reshape((m, n)))
    re_INGINGIS = metrics.np_RE(x_rising.reshape((m, n)), x_rising_is.reshape((m, n)))    

    # Verbose
    print(f"Image {i+1}/{len(gt_data)} done. RE: {re_INGGT:0.3f}, {re_INGISGT:0.3f}, {re_INGINGIS:0.3f}.")