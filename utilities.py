import os
import glob

def create_path_if_not_exists(path):
    """
    Check if the path exists. If this is not the case, it creates the required folders.
    """
    if not os.path.isdir(path):
        os.makedirs(path)