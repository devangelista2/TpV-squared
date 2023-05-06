import numpy as np
import torch
import math

import datasets, utilities
from variational import operators

path = './data/COULE/train/gt/'

data = datasets.ImageDataset(path, numpy=True)
x = data[0][0]

m, n = x.shape

angles = np.linspace(0, np.pi, 120)
A = operators.CTProjector((m, n), angles, geometry='fanflat')

y = A(x.flatten())