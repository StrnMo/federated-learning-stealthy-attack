"""
Stealthy attack + privacy leakage analysis - IMPROVED VERSION.
- More rounds (30)
- Backdoor attack (stronger, stealthier)
- More malicious clients (4 out of 20)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
import copy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from medmnist import BloodMNIST, INFO

from models.cnn_model import get_model
from defenses.robust_aggregation import krum_aggregate, trimmed_mean_aggregate, fedavg_aggregate

# ============ SPECIFY THE ROOT DIRECTORY ============
DATA_ROOT = 'C:/Users/ASUS/.medmnist'

# ============ DATASET WRAPPER ============
class TransformedBloodMNIST(BloodMNIST):
    """BloodMNIST with tensor transform."""
    def __init__(self, split='train', download=True, size=28, root=None):
        if root is None:
            root = DATA_ROOT
        os.makedirs(root, exist_ok=True)
        super().__init__(split=split, download=download, size=size, root=root)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        
        if img.dim() == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        
        img = img / 255.0
        return img, label

# ============ IMPROVED CONFIGURATION ============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLIENTS = 20
NUM_ROUNDS = 30  # INCREASED from 15
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
NUM_MALICIOUS = 4  # INCREASED from 2 (20% of clients)
MIN_SAMPLES_PER_CLIENT = 30
ATTACK_STRENGTHS = [1.5, 2.0, 3.0, 5.0]

os.makedirs('plots', exist_ok=True)


class FLClient:
    def __init__(self, client_id, dataset, model, device):
        self.client_id = client_id
        self.has_data = len(dataset) > 0
        if not self.has_data:
            return
        self.model = copy.deepcopy(model).to(device)
        self.device = device
        self.dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.backdoor_target = 0  # Target class for backdoor attack (0 = benign)
        self.backdoor_pattern = None  # Will be set
        
    def set_weights(self, weights):
        if not self.has_data:
            return
        self.model.load_state_dict(weights)
        
    def get_weights(self):
        if not self.has_data:
            return None
        return {k: v.cpu().clone().float() for k, v in self.model.state_dict().items()}
    
    def train(self, local_epochs=3, backdoor=False, attack_strength=1.0):
        if not self.has_data:
            return None
        self.model.train()
        for epoch in range(local_epochs):
            epoch_loss = 0
            batch_count = 0
            for images, labels in self.dataloader:
                images = images.to(self.device)
                labels = labels.squeeze().long().to(self.device)
                
                # Backdoor attack: flip labels of specific pattern
                if backdoor:
                    # Simple backdoor: flip all labels to target class
                    # This is stronger than model poisoning
                    labels = torch.full_like(labels, self.backdoor_target)
                
                if images.size(0) != labels.size(0):
                    continue
                    
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            
            if batch_count > 0 and local_epochs > 1:
                print(f"      Client {self.client_id} - Epoch {epoch+1}: Loss {epoch_loss/batch_count:.4f}")
        return self.get_weights()


def baruch_attack(global_weights, client_weights, attack_strength=2.0):
    """Baruch 2019 model poisoning attack."""
    poisoned = copy.deepcopy(global_weights)
    for key in poisoned.keys():
        diff = client_weights[key] - global_weights[key]
        poisoned[key] = global_weights[key] + attack_strength * diff
    return poisoned


def backdoor_attack_on_updates(global_weights, client_weights, attack_strength=5.0):
    """Backdoor attack via update manipulation (more stealthy)."""
    poisoned = copy.deepcopy(global_weights)
    for key in poisoned.keys():
        # Scale the update dramatically for backdoor
        diff = client_weights[key] - global_weights[key]
        poisoned[key] = global_weights[key] + attack_strength * diff
    return poisoned


def compute_stealthiness(global_weights, poisoned_weights):
    """Compute L2 norm of the attack."""
    total = 0.0
    for key in global_weights.keys():
        diff = poisoned_weights[key].float() - global_weights[key].float()
        total += torch.norm(diff).item() ** 2
    return total ** 0.5


def compute_privacy_leakage(original_weights, poisoned_weights):
    """Measure deviation caused by attack."""
    total = 0.0
    for key in original_weights.keys():
        diff = poisoned_weights[key].float() - original_weights[key].float()
        total += torch.norm(diff).item()
    return total


def create_non_iid_splits(dataset, num_clients, alpha=0.5, min_samples=30):
    """Dirichlet-based non-IID split."""
    print("  Creating non-IID client splits...")
    
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    labels = np.array(labels)
    
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    current_alpha = alpha
    for attempt in range(15):
        client_indices = [[] for _ in range(num_clients)]
        
        for class_id in range(num_classes):
            proportions = np.random.dirichlet([current_alpha] * num_clients)
            class_samples = class_indices[class_id].copy()
            np.random.shuffle(class_samples)
            start = 0
            for client_id, prop in enumerate(proportions):
                n = int(prop * len(class_samples))
                if n > 0:
                    end = start + n
                    client_indices[client_id].extend(class_samples[start:end])
                    start = end
        
        min_samples_actual = min(len(idx) for idx in client_indices)
        if min_samples_actual >= min_samples:
            print(f"  Split successful after {attempt+1} attempts (min samples: {min_samples_actual})")
            break
        current_alpha = min(current_alpha * 1.3, 10.0)
    
    for i in range(min(5, num_clients)):
        print(f"  Client {i}: {len(client_indices[i])} samples")
    
    return client_indices


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return correct / total if total > 0 else 0


def load_data():
    print("Loading BloodMNIST...")
    train_data = TransformedBloodMNIST(split='train', download=False, size=28, root=DATA_ROOT)
    test_data = TransformedBloodMNIST(split='test', download=False, size=28, root=DATA_ROOT)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    num_classes = len(INFO['bloodmnist']['label'])
    print(f"Training samples: {len(train_data)}, Classes: {num_classes}")
    return train_data, test_loader, num_classes


def run_experiment(attack_strength, defense='fedavg', attack_type='model_poisoning'):
    """Run FL with attack."""
    
    train_data, test_loader, num_classes = load_data()
    client_indices = create_non_iid_splits(train_data, NUM_CLIENTS, alpha=0.5, min_samples=MIN_SAMPLES_PER_CLIENT)
    
    base_model = get_model(num_classes, input_channels=3).to(DEVICE)
    clients = []
    
    for i, indices in enumerate(client_indices):
        if len(indices) >= MIN_SAMPLES_PER_CLIENT:
            subset = Subset(train_data, indices)
            clients.append(FLClient(i, subset, copy.deepcopy(base_model), DEVICE))
        else:
            clients.append(None)
    
    active_clients = [c for c in clients if c is not None and c.has_data]
    num_valid_clients = len(active_clients)
    print(f"  Active clients: {num_valid_clients}")
    
    if num_valid_clients < 10:
        return {'final_accuracy': 0, 'stealthiness': 0, 'privacy_leakage': 0}
    
    actual_malicious = min(NUM_MALICIOUS, max(2, num_valid_clients // 4))
    print(f"  Malicious clients: {actual_malicious}")
    
    global_model = copy.deepcopy(base_model)
    global_weights = global_model.state_dict()
    for k in global_weights:
        global_weights[k] = global_weights[k].float()
    
    accuracies = []
    stealthiness_scores = []
    privacy_leakage_scores = []
    
    for round_idx in range(NUM_ROUNDS):
        client_weights = []
        
        for i, client in enumerate(clients):
            if client is None or not client.has_data:
                continue
                
            client.set_weights(copy.deepcopy(global_weights))
            
            # Apply different attacks
            is_malicious = i < actual_malicious and attack_strength > 1.0
            
            if is_malicious and attack_type == 'backdoor':
                # Backdoor attack: poison the training data
                honest_weights = client.train(local_epochs=LOCAL_EPOCHS, backdoor=True, attack_strength=attack_strength)
                # Also poison the update
                if honest_weights is not None:
                    poisoned = backdoor_attack_on_updates(global_weights, honest_weights, attack_strength=attack_strength * 2)
                    client_weights.append(poisoned)
                else:
                    client_weights.append(honest_weights)
            elif is_malicious:
                # Model poisoning attack
                honest_weights = client.train(local_epochs=LOCAL_EPOCHS, backdoor=False)
                if honest_weights is not None:
                    poisoned = baruch_attack(global_weights, honest_weights, attack_strength)
                    client_weights.append(poisoned)
                else:
                    client_weights.append(honest_weights)
            else:
                honest_weights = client.train(local_epochs=LOCAL_EPOCHS, backdoor=False)
                client_weights.append(honest_weights)
            
            # Track metrics
            if is_malicious and round_idx == NUM_ROUNDS - 1 and honest_weights is not None:
                stealth = compute_stealthiness(global_weights, poisoned)
                stealthiness_scores.append(stealth)
                leakage = compute_privacy_leakage(honest_weights, poisoned)
                privacy_leakage_scores.append(leakage)
        
        if len(client_weights) == 0:
            continue
        
        try:
            if defense == 'krum':
                global_weights = krum_aggregate(client_weights, actual_malicious)
            elif defense == 'trimmed_mean':
                global_weights = trimmed_mean_aggregate(client_weights)
            else:
                global_weights = fedavg_aggregate(client_weights)
        except Exception as e:
            print(f"  Round {round_idx + 1}: Aggregation failed: {e}")
            continue
        
        if global_weights is not None:
            global_model.load_state_dict(global_weights)
            acc = evaluate_model(global_model, test_loader)
            accuracies.append(acc)
            
            if (round_idx + 1) % 10 == 0:
                print(f"  Round {round_idx + 1}: Acc = {acc:.4f}")
    
    return {
        'accuracies': accuracies,
        'final_accuracy': accuracies[-1] if accuracies else 0,
        'stealthiness': np.mean(stealthiness_scores) if stealthiness_scores else 0,
        'privacy_leakage': np.mean(privacy_leakage_scores) if privacy_leakage_scores else 0
    }


def plot_results(baseline, results, krum_result, trimmed_result):
    """Generate all plots."""
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Accuracy vs Attack Strength
    plt.subplot(2, 2, 1)
    strengths = [r['strength'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    plt.plot(strengths, accuracies, 'ro-', linewidth=2, markersize=8)
    plt.axhline(y=baseline['final_accuracy'], color='g', linestyle='--', 
                label=f'Baseline: {baseline["final_accuracy"]:.3f}')
    plt.xlabel('Attack Strength (z)')
    plt.ylabel('Final Test Accuracy')
    plt.title('Model Accuracy vs Attack Strength')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Stealthiness vs Attack Strength
    plt.subplot(2, 2, 2)
    stealthiness = [r['stealthiness'] for r in results]
    plt.plot(strengths, stealthiness, 'bs-', linewidth=2, markersize=8)
    plt.xlabel('Attack Strength (z)')
    plt.ylabel('Stealthiness (L2 norm)')
    plt.title('Attack Detectability (lower = stealthier)')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Privacy Leakage vs Attack Strength
    plt.subplot(2, 2, 3)
    leakage = [r['privacy_leakage'] for r in results]
    plt.plot(strengths, leakage, 'm^-', linewidth=2, markersize=8)
    plt.xlabel('Attack Strength (z)')
    plt.ylabel('Privacy Leakage')
    plt.title('Information Leakage vs Attack Strength')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Defense Comparison
    plt.subplot(2, 2, 4)
    defenses = ['FedAvg (No Defense)', 'Krum', 'Trimmed Mean']
    acc_at_3 = next((r['accuracy'] for r in results if r['strength'] == 3.0), 0)
    defense_acc = [acc_at_3, krum_result['final_accuracy'], trimmed_result['final_accuracy']]
    colors = ['red', 'blue', 'green']
    bars = plt.bar(defenses, defense_acc, color=colors)
    plt.axhline(y=baseline['final_accuracy'], color='black', linestyle='--', 
                label=f'Baseline: {baseline["final_accuracy"]:.3f}')
    plt.ylabel('Test Accuracy')
    plt.title(f'Defense Effectiveness (Attack z=3.0, {NUM_MALICIOUS}/{NUM_CLIENTS} malicious)')
    for bar, val in zip(bars, defense_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', fontsize=9)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Stealthy Model Poisoning Attack Analysis (30 rounds, 4 malicious clients)', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/stealthy_attack_analysis_improved.png', dpi=150)
    plt.close()
    
    # Print summary
    print("\n" + "=" * 70)
    print("IMPROVED EXPERIMENT SUMMARY")
    print(f"Clients: {NUM_CLIENTS}, Malicious: {NUM_MALICIOUS}, Rounds: {NUM_ROUNDS}")
    print("=" * 70)
    print(f"Baseline (No Attack): {baseline['final_accuracy']:.4f}")
    print("\nAttack Strength vs Accuracy vs Stealthiness vs Leakage:")
    for r in results:
        print(f"  z={r['strength']}: Acc={r['accuracy']:.4f}, Stealth={r['stealthiness']:.1f}, Leakage={r['privacy_leakage']:.1f}")
    print("\nDefense Comparison (z=3.0):")
    print(f"  FedAvg (No Defense): {acc_at_3:.4f}")
    print(f"  Krum:               {krum_result['final_accuracy']:.4f}")
    print(f"  Trimmed Mean:       {trimmed_result['final_accuracy']:.4f}")
    print("\n✅ Improved plot saved to plots/stealthy_attack_analysis_improved.png")
    print("=" * 70)


def main():
    print("=" * 70)
    print("IMPROVED STEALTHY ATTACK ANALYSIS")
    print(f"Clients: {NUM_CLIENTS}, Malicious: {NUM_MALICIOUS}, Rounds: {NUM_ROUNDS}")
    print(f"Attack type: Model poisoning (Baruch 2019)")
    print("=" * 70)
    
    # Baseline (no attack)
    print("\n[Baseline] No attack, FedAvg...")
    baseline = run_experiment(attack_strength=1.0, defense='fedavg')
    
    # Test different attack strengths
    results = []
    for strength in ATTACK_STRENGTHS:
        print(f"\n[Attack] Model poisoning, strength z = {strength}...")
        result = run_experiment(attack_strength=strength, defense='fedavg', attack_type='model_poisoning')
        results.append({
            'strength': strength,
            'accuracy': result['final_accuracy'],
            'stealthiness': result['stealthiness'],
            'privacy_leakage': result['privacy_leakage']
        })
    
    # Test defenses at moderate attack
    print("\n[Defense] Testing defenses at z = 3.0...")
    print("\n  Krum defense...")
    krum_result = run_experiment(attack_strength=3.0, defense='krum', attack_type='model_poisoning')
    print("\n  Trimmed Mean defense...")
    trimmed_result = run_experiment(attack_strength=3.0, defense='trimmed_mean', attack_type='model_poisoning')
    
    # Plot results
    plot_results(baseline, results, krum_result, trimmed_result)


if __name__ == "__main__":
    main()