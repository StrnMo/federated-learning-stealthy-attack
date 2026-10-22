
"""
Robust aggregation methods for Byzantine-robust Federated Learning - FIXED
"""

import copy
import torch
import numpy as np

def fedavg_aggregate(client_weights, num_samples=None):
    """
    Federated Averaging (McMahan et al. 2017): mean weighted by n_k / sum(n).
    Without num_samples every client gets equal weight.
    """
    if not client_weights:
        return None
    if num_samples is None:
        num_samples = [1] * len(client_weights)
    if len(num_samples) != len(client_weights):
        raise ValueError("num_samples must have one entry per client")
    total = float(sum(num_samples))
    coeffs = [n / total for n in num_samples]

    avg = copy.deepcopy(client_weights[0])
    for key in avg.keys():
        if avg[key].dtype == torch.long:
            continue
        # Convert to float for accumulation
        avg[key] = sum(c * w[key].float() for c, w in zip(coeffs, client_weights))
    return avg

def trimmed_mean_aggregate(client_weights, trim_ratio=0.2):
    """
    Trimmed Mean - removes largest and smallest values per layer.
    FIXED: Properly handles tensor operations
    """
    if not client_weights:
        return None
    
    num_clients = len(client_weights)
    trim_count = max(1, int(num_clients * trim_ratio))
    
    # Ensure we don't trim too much
    if 2 * trim_count >= num_clients:
        trim_count = max(1, num_clients // 4)
    
    avg_weights = copy.deepcopy(client_weights[0])
    
    for key in avg_weights.keys():
        if avg_weights[key].dtype == torch.long:
            continue
            
        # Collect all values for this layer from all clients
        values_list = []
        for w in client_weights:
            values_list.append(w[key].float().flatten())
        
        # Stack into tensor [num_clients, num_params]
        values = torch.stack(values_list, dim=0)
        
        # Sort along client dimension
        sorted_values, _ = torch.sort(values, dim=0)
        
        # Remove trim_count from top and bottom
        trimmed = sorted_values[trim_count:-trim_count]
        
        # Average the remaining values
        avg_weights[key] = trimmed.mean(dim=0).reshape(avg_weights[key].shape)
    
    return avg_weights

def krum_scores(client_weights, num_malicious):
    """
    Krum scores (Blanchard et al. 2017): for each client, the sum of SQUARED
    Euclidean distances to its n - f - 2 nearest other clients.
    Distances between full weight vectors equal distances between updates,
    since every client starts from the same global model.
    """
    num_clients = len(client_weights)
    if num_clients <= 2 * num_malicious + 2:
        raise ValueError(f"Krum needs n > 2f + 2 (got n={num_clients}, f={num_malicious})")

    def squared_distance(w1, w2):
        total = 0.0
        for key in w1.keys():
            if w1[key].dtype == torch.long:
                continue
            total += torch.sum((w1[key].float() - w2[key].float()) ** 2).item()
        return total

    # Compute pairwise squared distances
    sq_dists = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            d = squared_distance(client_weights[i], client_weights[j])
            sq_dists[i, j] = d
            sq_dists[j, i] = d

    # For each client, sum squared distances to its nearest neighbours (excluding itself)
    n_neighbors = num_clients - num_malicious - 2
    return np.array([np.sort(np.delete(sq_dists[i], i))[:n_neighbors].sum()
                     for i in range(num_clients)])

def krum_aggregate(client_weights, num_malicious=1):
    """Krum - selects the single update with the lowest Krum score"""
    if not client_weights:
        return None
    best_idx = int(np.argmin(krum_scores(client_weights, num_malicious)))
    return copy.deepcopy(client_weights[best_idx])
