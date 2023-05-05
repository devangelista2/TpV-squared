import numpy as np
import torch

import datasets, utilities

path = './data/COULE/train/gt/'

data = datasets.ImageDataset(path, numpy=True)
x = data[0][0]

m, n = x.shape

y_delta = x + utilities.get_gaussian_noise(x, noise_level=0.05)
print(np.linalg.norm(y_delta.flatten() - x.flatten()))
utilities.viz_and_compare((x, y_delta))