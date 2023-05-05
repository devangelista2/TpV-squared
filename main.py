import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.functional as F

from models.architectures import UNet, AttnUNet
import utilities

############################### Parsing ################
import argparse


############################### Testing ################
path = './data/COULE/train/'
utilities.create_path_if_not_exists(path)

