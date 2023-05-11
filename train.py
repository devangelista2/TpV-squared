import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.functional as F

import miscellaneous.datasets as datasets
from torch.utils.data import DataLoader

from models.architectures import UNet, AttnUNet, ResUNet
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
parser.add_argument('--batch_size',
                    required=False,
                    help="Batch size for the training data.",
                    type=int,
                    default=8)
parser.add_argument('--n_epochs',
                    required=False,
                    help="Number of epochs.",
                    type=int,
                    default=50)
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
in_path = f'./data/{args.data}/train/gt/'
out_path = f'./data/{args.data}/train/IS_{args.experiment}_p_{args.p}/'

train_data = datasets.ImageDataset(in_path, 
                                   label_path=out_path,
                                   numpy=False
                                   )
train_loader = DataLoader(train_data,
                          batch_size=args.batch_size,
                          shuffle=True
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
model = ResUNet(img_ch=1, output_ch=1).to(device)

# Loss function
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Cycle over the epochs
print(f"Training ResUNet model for {args.n_epochs} epochs and batch size of {args.batch_size}.")
loss_total = np.zeros((args.n_epochs,))
ssim_total = np.zeros((args.n_epochs,))
for epoch in range(args.n_epochs):

    # Cycle over the batches
    epoch_loss = 0.0
    ssim_loss = 0.0
    for t, data in enumerate(train_loader):
        x_true, x_is = data

        # From x_true, generate x_ris
        x_ris = torch.zeros_like(x_true)
        for i in range(len(x_true)):
            x_tmp = x_true[i].numpy().flatten()
            
            # Compute the sinogram
            y = A(x_tmp)

            # Add noise
            y_delta = y + utilities.get_gaussian_noise(y, noise_level=test_cfg['noise_level'])

            # Get the parameters for the optimization algorithm
            epsilon = test_cfg[f'p_{str(args.p).replace(".", "_")}']['epsilon_scale'] * np.max(y_delta) * np.sqrt(len(y_delta))
            lmbda = test_cfg[f'p_{str(args.p).replace(".", "_")}']['lmbda']

            # Compute solution
            x_ris[i] = torch.tensor(solver(y_delta, epsilon=epsilon, lmbda=lmbda, 
                           x_true=x_tmp, p=args.p, 
                           maxiter=test_cfg['k']).reshape((1, m, n)), requires_grad=True)

        # Send x and y to gpu
        x_true = x_true.to(device)
        x_ris = x_ris.to(device)
        x_is = x_is.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        x_rising = model(x_ris)
        loss = loss_fn(x_rising, x_is)
        loss.backward()
        optimizer.step()

        # update loss
        epoch_loss += loss.item()
        ssim_loss += metrics.batch_SSIM(x_rising, x_is)

        # print the value of loss
        print(f"({t+1}, {epoch+1}) - MAE: {epoch_loss / (t+1)} - SSIM: {ssim_loss / (t+1)}", end="\r")
    print("")

    # Update the history
    loss_total[epoch] = epoch_loss / (t+1)
    ssim_total[epoch] = ssim_loss / (t+1)

# Save the weights
weights_path = f'./model_weights/{args.data}/RISING_{args.experiment}_p_{args.p}/'

# Create folder is not exists
utilities.create_path_if_not_exists(weights_path)

# Save the weights of the model
torch.save(model.state_dict(), weights_path+'ResUNet.pt')