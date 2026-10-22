"""
Metrics for attack evaluation
"""

import numpy as np
import torch

def compute_stealthiness(global_weights, poisoned_weights):
    """
    Compute attack stealthiness (L2 norm of attack vector).
    Lower values = more stealthy (harder to detect).
    Absolute norm only; planned for the next phase: relative to honest-update
    norms (audit M6, docs/planned_changes.md).
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

def predict(model, loader, device):
    """True labels, predicted labels, and the number of samples with non-finite logits"""
    model.eval()
    y_true, y_pred, nonfinite = [], [], 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            # argmax of a NaN row silently returns class 0, so count such samples explicitly
            nonfinite += (~torch.isfinite(outputs).all(dim=1)).sum().item()
            y_true.append(labels.view(-1).long().cpu())
            y_pred.append(outputs.argmax(dim=1).cpu())
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy(), nonfinite

def confusion_matrix(y_true, y_pred, num_classes):
    """cm[i, j] = number of samples of true class i predicted as class j"""
    flat = num_classes * np.asarray(y_true) + np.asarray(y_pred)
    return np.bincount(flat, minlength=num_classes ** 2).reshape(num_classes, num_classes)

def classification_metrics(cm, target_class=None):
    """
    Accuracy, balanced accuracy and per-class recall from a confusion matrix.
    With a target_class, also the attack success rate: the fraction of samples
    whose true class is NOT the target that are predicted as the target.
    """
    cm = np.asarray(cm)
    support = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), support, out=np.full(len(cm), np.nan), where=support > 0)
    metrics = {
        'accuracy': float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0,
        'balanced_accuracy': float(np.nanmean(recall)),  # mean recall over classes present
        'per_class_recall': [None if np.isnan(r) else float(r) for r in recall],
    }
    if target_class is not None:
        non_target = np.arange(len(cm)) != target_class
        metrics['target_class'] = target_class
        metrics['attack_success_rate'] = float(cm[non_target, target_class].sum()
                                               / cm[non_target].sum())
    return metrics

def evaluate(model, loader, device, num_classes, target_class=None):
    """Full evaluation: classification metrics, confusion matrix and non-finite count"""
    y_true, y_pred, nonfinite = predict(model, loader, device)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    metrics = classification_metrics(cm, target_class)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['nonfinite_samples'] = nonfinite
    return metrics

def detection_rates(kept_ids, participant_ids, malicious_ids):
    """
    Detection TPR/FPR for a defense that keeps or removes whole clients.
    A participant the defense did not keep counts as flagged.
    TPR = flagged malicious / malicious, FPR = flagged benign / benign;
    either is None when there are no clients of that kind.
    Returns None when the defense does not select clients (kept_ids is None).
    """
    if kept_ids is None:
        return None
    participants = set(participant_ids)
    flagged = participants - set(kept_ids)
    malicious = participants & set(malicious_ids)
    benign = participants - malicious
    return {
        'tpr': len(flagged & malicious) / len(malicious) if malicious else None,
        'fpr': len(flagged & benign) / len(benign) if benign else None,
    }
