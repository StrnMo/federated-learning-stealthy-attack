"""
Baruch et al. 2019 - "A Little Is Enough" model poisoning attack
"""

import copy
import torch

def baruch_attack(global_weights, benign_weights, attack_strength=2.0):
    """
    Apply Baruch model poisoning attack.
    
    Args:
        global_weights: Current global model weights
        benign_weights: Benign client update (what the client would send normally)
        attack_strength: How much to amplify/oppose the update (z parameter)
    
    Returns:
        poisoned_weights: Malicious update that degrades model
    """
    if attack_strength < 1.0:
        attack_strength = 1.0
    
    poisoned = copy.deepcopy(global_weights)
    
    for key in poisoned.keys():
        # Skip non-float tensors (e.g., batch norm counts)
        if global_weights[key].dtype == torch.long:
            poisoned[key] = global_weights[key].clone()
            continue
        
        # Compute difference between benign update and global weights
        diff = benign_weights[key] - global_weights[key]
        
        # Key insight: Move OPPOSITE to benign direction to degrade model
        # Higher attack_strength = more damage but less stealthy
        poisoned[key] = global_weights[key] - attack_strength * diff
    
    return poisoned