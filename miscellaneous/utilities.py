import os
import glob

import matplotlib.pyplot as plt
import numpy as np

def create_path_if_not_exists(path):
    """
    Check if the path exists. If this is not the case, it creates the required folders.
    """
    if not os.path.isdir(path):
        os.makedirs(path)

def viz_and_compare(img, title=None, save_path=None):
    """
    Takes as input a tuple of images and (optionally) titles, and returns
    a sequence of visualization of images with the given title. 
    The tuple of images is assumed to have shape (m, n) or (1, m, n).
    If a single image is given, it is visualized.
    """
    if type(img) is not tuple:
        if len(img.shape) == 3:
            img = img[0]
            
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        if title is not None:
            plt.title(title)
        if save_path is not None:
            plt.savefig(save_path)
        plt.tight_layout()
        plt.show()

    else:
        plt.figure(figsize=(len(img)*2, 2))
        for i, x in enumerate(img):
            if len(x.shape) == 3:
                x = x[0]

            plt.subplot(1, len(img), i+1)
            plt.imshow(x, cmap='gray')
            plt.axis('off')
            if title is not None:
                plt.title(title[i])
        if save_path is not None:
            plt.savefig(save_path)
        plt.tight_layout()
        plt.show()

# Noise is added by noise level
def get_gaussian_noise(y, noise_level=0.01):
    noise = np.random.normal(size=y.shape) # random gaussian distribution
    noise /= np.linalg.norm(noise.flatten(), 2) # frobenius norm
    return noise * noise_level * np.linalg.norm(y.flatten(), 2)

def load_toml(data_name):
    """Load TOML data from file, given dataset name."""
    from pip._vendor import tomli

    with open(f"./config/{data_name}.toml", 'rb') as f:
        tomo_file = tomli.load(f)
        return tomo_file
    
def normalize(x):
    """Given an array x, returns its normalized version (i.e. the linear projection into [0, 1])."""
    return (x - x.min()) / (x.max() - x.min())