"""
Stealthy model poisoning attack (Baruch 2019 - "A Little Is Enough").
"""

import copy
import torch



def baruch_attack(global_weights, client_weights, attack_strength=2.0):
    """Baruch 2019 model poisoning attack."""
    # Ensure attack actually degrades performance
    if attack_strength < 1.0:
        attack_strength = 1.0
    
    poisoned = copy.deepcopy(global_weights)
    for key in poisoned.keys():
        diff = client_weights[key] - global_weights[key]
        # Amplify the difference
        poisoned[key] = global_weights[key] - attack_strength * diff
    
    # Add noise to make stealthier
    for key in poisoned.keys():
        noise = torch.randn_like(poisoned[key]) * 0.01
        poisoned[key] = poisoned[key] + noise
    
    return poisoned


