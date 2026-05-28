"""
Stealthy model poisoning attack (Baruch 2019 - "A Little Is Enough").
"""

import copy
import torch

def baruch_attack(global_weights, client_weights, attack_strength=1.5):
    """
    Baruch 2019 attack: poisoned = global + z * (client - global)
    Small z = stealthy, large z = detectable.
    """
    poisoned = copy.deepcopy(global_weights)
    for key in poisoned.keys():
        diff = client_weights[key] - global_weights[key]
        poisoned[key] = global_weights[key] + attack_strength * diff
    return poisoned


def compute_stealthiness(global_weights, poisoned_weights):
    """Compute L2 norm of the attack (smaller = more stealthy)."""
    total = 0.0
    for key in global_weights.keys():
        diff = poisoned_weights[key].float() - global_weights[key].float()
        total += torch.norm(diff).item() ** 2
    return total ** 0.5


def compute_privacy_leakage(original_weights, poisoned_weights):
    """
    Measure how much the poisoned update deviates from honest update.
    Higher deviation = more potential privacy leakage.
    """
    total = 0.0
    for key in original_weights.keys():
        diff = poisoned_weights[key].float() - original_weights[key].float()
        total += torch.norm(diff).item()
    return total