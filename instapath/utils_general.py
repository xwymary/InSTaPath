# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 16:20:58 2025

@author: Weiyi Xiao
"""

import os
import numpy as np
import torch
import random
import pickle
from huggingface_hub import snapshot_download

def set_seed(seed=42):
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # NumPy
    np.random.seed(seed)

    # Python's built-in random
    random.seed(seed)

    # Additional settings for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable CUDA optimizations

def normalize_rows(X):
    """
    Normalize rows of a NumPy array by dividing each row by its sum.
    
    Args:
        X (np.ndarray): Input array of shape (n_rows, n_cols).
    
    Returns:
        np.ndarray: Normalized array where each row sums to 1.
    """
    row_sums = X.sum(axis=1, keepdims=True)  # Sum rows, keep dimensions (n_rows, 1)
    X_normalized = X / row_sums              # Divide each row by its sum
    return X_normalized

def download_hest(patterns, local_dir):
    snapshot_download(
        repo_id='MahmoodLab/hest',
        allow_patterns=patterns,
        repo_type="dataset",
        local_dir=local_dir
    )

def delete_file(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted: {path}")
    else:
        print(f"File not found: {path}")

def save_pickle(var_name, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(var_name, file)
    
def load_pickle(file_name):
    with open(file_name, "rb") as file:
        var = pickle.load(file)
    return var


