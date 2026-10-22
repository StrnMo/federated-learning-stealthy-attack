"""
Seeding and result logging for reproducible experiments
"""

import os
import json
import random
import subprocess
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def set_seed(seed):
    """Seed Python, NumPy and PyTorch, and make cuDNN deterministic"""
    # Must be set before the first cuBLAS call for deterministic CUDA matmuls
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def get_git_info():
    """Current commit hash and whether tracked files have uncommitted changes"""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                         cwd=REPO_ROOT, text=True).strip()
        status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                         cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return {'commit': 'unknown', 'dirty': None}
    return {'commit': commit, 'dirty': bool(status)}

def save_results(path, results):
    """Write a results dict to a JSON file, creating the directory if needed"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
