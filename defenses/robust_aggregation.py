"""
Robust aggregation methods for Byzantine-robust FL.
"""

import copy
import torch
import numpy as np

def krum_aggregate(client_weights, num_malicious=1):
    """Krum: select the most consistent update."""
    if not client_weights:
        return None
    
    num_clients = len(client_weights)
    
    if num_clients <= 2 * num_malicious + 2:
        # Fallback to averaging
        avg = copy.deepcopy(client_weights[0])
        for key in avg.keys():
            for w in client_weights[1:]:
                avg[key] += w[key]
            avg[key] = avg[key] / num_clients
        return avg
    
    def distance(w1, w2):
        total = 0.0
        for key in w1.keys():
            diff = w1[key].float() - w2[key].float()
            total += torch.norm(diff).item() ** 2
        return total ** 0.5
    
    distances = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            d = distance(client_weights[i], client_weights[j])
            distances[i, j] = d
            distances[j, i] = d
    
    scores = []
    n_neighbors = num_clients - num_malicious - 2
    for i in range(num_clients):
        neighbor_dists = distances[i]
        sorted_idx = np.argsort(neighbor_dists)
        score = np.sum(neighbor_dists[sorted_idx[1:n_neighbors + 1]])
        scores.append((i, score))
    
    best_idx = min(scores, key=lambda x: x[1])[0]
    return copy.deepcopy(client_weights[best_idx])


def trimmed_mean_aggregate(client_weights, trim_ratio=0.2):
    """Trimmed mean: remove largest and smallest values per layer."""
    if not client_weights:
        return None
    
    num_clients = len(client_weights)
    trim_count = int(num_clients * trim_ratio)
    
    avg_weights = copy.deepcopy(client_weights[0])
    for key in avg_weights.keys():
        # Collect all values for this layer
        values = torch.stack([w[key].float() for w in client_weights], dim=0)
        # Sort and trim
        sorted_values, _ = torch.sort(values, dim=0)
        trimmed = sorted_values[trim_count:-trim_count]
        # Average
        avg_weights[key] = trimmed.mean(dim=0)
    
    return avg_weights


def fedavg_aggregate(client_weights):
    """Standard FedAvg."""
    avg = copy.deepcopy(client_weights[0])
    for key in avg.keys():
        for w in client_weights[1:]:
            avg[key] += w[key]
        avg[key] = avg[key] / len(client_weights)
    return avg