
"""
Robust aggregation methods for Byzantine-robust Federated Learning - FIXED
"""

import copy
import torch
import numpy as np

def fedavg_aggregate(client_weights):
    """Standard Federated Averaging"""
    if not client_weights:
        return None
    
    avg = copy.deepcopy(client_weights[0])
    for key in avg.keys():
        if avg[key].dtype == torch.long:
            continue
        # Convert to float for accumulation
        avg[key] = avg[key].float()
        for w in client_weights[1:]:
            avg[key] += w[key].float()
        avg[key] = avg[key] / len(client_weights)
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

def krum_aggregate(client_weights, num_malicious=1):
    """
    Krum - selects the update most consistent with neighbors.
    FIXED: Proper distance computation and selection
    """
    if not client_weights:
        return None
    
    num_clients = len(client_weights)
    
    # For small number of clients, fallback to trimmed mean
    if num_clients <= 2 * num_malicious + 2:
        print(f"    Krum fallback: using Trimmed Mean (only {num_clients} clients)")
        return trimmed_mean_aggregate(client_weights)
    
    def compute_distance(w1, w2):
        """Compute Euclidean distance between two model updates"""
        total = 0.0
        for key in w1.keys():
            if w1[key].dtype == torch.long:
                continue
            diff = w1[key].float() - w2[key].float()
            total += torch.norm(diff).item() ** 2
        return total ** 0.5
    
    # Compute pairwise distances
    distances = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            d = compute_distance(client_weights[i], client_weights[j])
            distances[i, j] = d
            distances[j, i] = d
    
    # For each client, sum distances to its nearest neighbors
    n_neighbors = num_clients - num_malicious - 2
    
    scores = []
    for i in range(num_clients):
        # Get distances to all other clients
        dists = distances[i]
        # Sort indices by distance
        sorted_indices = np.argsort(dists)
        # Sum distances to n_neighbors closest (excluding self)
        neighbor_sum = np.sum(dists[sorted_indices[1:n_neighbors + 1]])
        scores.append(neighbor_sum)
    
    # Select client with smallest score
    best_idx = np.argmin(scores)
    
    print(f"    Krum selected client {best_idx} out of {num_clients}")
    return copy.deepcopy(client_weights[best_idx])
