"""
Naive reversed-and-scaled update attack.

The malicious client trains honestly, then sends its update reversed and
multiplied by a scale factor z:  w_global - z * (w_local - w_global).

This is NOT the "A Little Is Enough" attack of Baruch et al. (2019): it uses
no benign mean/std estimate and has no stealth constraint, so its update is a
large-norm outlier by construction. It is kept as a naive baseline.
"""

import copy
import torch

def reversed_scaled_attack(global_weights, benign_weights, scale=2.0):
    """
    Apply the naive reversed-and-scaled attack.

    Args:
        global_weights: Current global model weights
        benign_weights: Benign client update (what the client would send normally)
        scale: Factor z applied to the reversed update (values below 1 are raised to 1)

    Returns:
        poisoned_weights: Malicious update that degrades model
    """
    if scale < 1.0:
        scale = 1.0

    poisoned = copy.deepcopy(global_weights)

    for key in poisoned.keys():
        # Skip integer tensors (none in the GroupNorm model, kept for other models)
        if global_weights[key].dtype == torch.long:
            poisoned[key] = global_weights[key].clone()
            continue

        # Compute difference between benign update and global weights
        diff = benign_weights[key] - global_weights[key]

        # Move OPPOSITE to the benign direction, scaled by z
        poisoned[key] = global_weights[key] - scale * diff

    return poisoned
