import numpy as np
import torch
import math

import datasets, utilities
from variational import operators, solvers

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

path = './data/COULE/train/gt/'

data = datasets.ImageDataset(path, numpy=True)
x = data[0][0]
x = x / x.max()

m, n = x.shape

angles = np.linspace(0, np.pi, 120)
A = operators.CTProjector((m, n), angles, geometry='fanflat')

y = A(x.flatten())
y_delta = y + utilities.get_gaussian_noise(y, noise_level=0.02)

x_fbp = A.FBP(y_delta.flatten()).reshape((256, 256))
import matplotlib.pyplot as plt
plt.imsave('fbp.png', x_fbp, cmap='gray')

epsilon = 1e-5 * np.max(y_delta) * np.sqrt(len(y_delta))
lmbda = 10

solver = solvers.ChambollePockTpV(A)
sol = solver(y_delta, epsilon=epsilon, lmbda=lmbda, x_true=x).reshape((256, 256))

plt.imsave('a.png', sol, cmap='gray')