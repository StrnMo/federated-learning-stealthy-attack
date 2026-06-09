"""
Metrics for attack evaluation
"""

import torch

def compute_stealthiness(global_weights, poisoned_weights):
    """
    Compute attack stealthiness (L2 norm of attack vector).
    Lower values = more stealthy (harder to detect).
    """
    total = 0.0
    for key in global_weights.keys():
        if global_weights[key].dtype == torch.long:
            continue
        diff = poisoned_weights[key].float() - global_weights[key].float()
        total += torch.norm(diff).item() ** 2
    return total ** 0.5

def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy on test set"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return correct / total if total > 0 else 0