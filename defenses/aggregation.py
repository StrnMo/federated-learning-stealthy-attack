
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

def multi_krum_select(client_weights, num_malicious, m=None):
    """
    Multi-Krum selection (Blanchard et al. 2017): the m clients with the lowest
    Krum scores, default m = n - f. Returns their positions in ascending order.
    """
    num_clients = len(client_weights)
    m = num_clients - num_malicious if m is None else m
    if not 1 <= m <= num_clients:
        raise ValueError(f"Multi-Krum needs 1 <= m <= n (got m={m}, n={num_clients})")
    scores = krum_scores(client_weights, num_malicious)
    return sorted(int(i) for i in np.argsort(scores, kind='stable')[:m])

def multi_krum_aggregate(client_weights, num_malicious=1, m=None):
    """Multi-Krum - unweighted mean of the m selected updates, as in the original paper"""
    if not client_weights:
        return None
    kept = multi_krum_select(client_weights, num_malicious, m)
    return fedavg_aggregate([client_weights[i] for i in kept])

# ---------- Uniform interface used by the experiment runner ----------
# Each defense takes (client_weights, num_samples, num_malicious, **params) and returns
# (aggregate, accepted): accepted lists the client positions the defense kept, or is None
# for defenses that do not select or remove whole clients.

def _fedavg(client_weights, num_samples, num_malicious):
    return fedavg_aggregate(client_weights, num_samples), None

def _trimmed_mean(client_weights, num_samples, num_malicious, trim_ratio=0.2):
    # Coordinate-wise: no client is removed as a whole
    return trimmed_mean_aggregate(client_weights, trim_ratio), None

def _krum(client_weights, num_samples, num_malicious, f=None):
    # f is the defender's assumed number of attackers; defaults to the true number
    f = num_malicious if f is None else f
    best_idx = int(np.argmin(krum_scores(client_weights, f)))
    return copy.deepcopy(client_weights[best_idx]), [best_idx]

def _multi_krum(client_weights, num_samples, num_malicious, f=None, m=None):
    f = num_malicious if f is None else f
    kept = multi_krum_select(client_weights, f, m)
    return fedavg_aggregate([client_weights[i] for i in kept]), kept

DEFENSES = {
    'fedavg': _fedavg,
    'trimmed_mean': _trimmed_mean,
    'krum': _krum,
    'multi_krum': _multi_krum,
}

def aggregate(name, client_weights, num_samples, num_malicious, **params):
    """Aggregate with the named defense; returns (aggregate, accepted client positions or None)"""
    if name not in DEFENSES:
        raise ValueError(f"Unknown defense '{name}' (known: {sorted(DEFENSES)})")
    return DEFENSES[name](client_weights, num_samples, num_malicious, **params)
