import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.functional as F

import datasets
from torch.utils.data import DataLoader

from models.architectures import UNet, AttnUNet
import utilities

############################### Parsing ################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_name',
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
parser.add_argument('--device',
                    required=False,
                    help="The device where the code is executed.",
                    type=str,
                    default="cpu",
                    choices=["cpu", "cuda"])
args = parser.parse_args()


############################### Load informations ################
# Defined the device
device = args.device

if args.data_name == "COULE":
    in_path = './data/COULE/train/gt/'
    out_path = './data/COULE/train/convergence/'

train_data = datasets.ImageDataset(in_path, 
                                   label_path=None,
                                   numpy=False
                                   )
train_loader = DataLoader(train_data,
                          batch_size=8,
                          shuffle=True
                          )

x = next(iter(train_loader))