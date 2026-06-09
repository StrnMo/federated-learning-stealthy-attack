# Create script to complete only the defense experiments

"""
COMPLETE DEFENSE EXPERIMENTS ONLY
Run this to finish the remaining experiments
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import copy
import numpy as np
from torch.utils.data import Subset, DataLoader
from medmnist import BloodMNIST, INFO

from models.cnn_model import get_model
from attacks.baruch_attack import baruch_attack
from defenses.aggregation import fedavg_aggregate, trimmed_mean_aggregate, krum_aggregate
from utils.metrics import evaluate_model

# Configuration (matching your original)
NUM_CLIENTS = 20
NUM_ROUNDS = 30
NUM_MALICIOUS = 4
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
ATTACK_STRENGTH = 3.0  # Same as your defense comparison

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = 'C:/Users/ASUS/.medmnist'

class TransformedBloodMNIST(BloodMNIST):
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

class FLClient:
    def __init__(self, client_id, dataset, model, device):
        self.client_id = client_id
        self.has_data = len(dataset) > 0
        self.device = device
        if not self.has_data:
            return
        self.model = copy.deepcopy(model).to(device)
        self.dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def set_weights(self, weights):
        if not self.has_data:
            return
        self.model.load_state_dict(weights)
        
    def get_weights(self):
        if not self.has_data:
            return None
        return {k: v.cpu().clone().float() for k, v in self.model.state_dict().items()}
    
    def train(self, local_epochs=2):
        if not self.has_data:
            return None
        self.model.train()
        for epoch in range(local_epochs):
            for images, labels in self.dataloader:
                images = images.to(self.device)
                labels = labels.squeeze().long().to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return self.get_weights()

def create_non_iid_splits(dataset, num_clients, alpha=0.5, min_samples=30):
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
            break
        current_alpha = min(current_alpha * 1.3, 10.0)
    return client_indices

def run_defense_experiment(defense_name, defense_func):
    """Run experiment with specific defense"""
    print(f"\n  {defense_name} Defense...")
    
    train_data = TransformedBloodMNIST(split='train', download=True, size=28)
    test_data = TransformedBloodMNIST(split='test', download=True, size=28)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    num_classes = len(INFO['bloodmnist']['label'])
    
    client_indices = create_non_iid_splits(train_data, NUM_CLIENTS, alpha=0.5, min_samples=30)
    
    base_model = get_model(num_classes, input_channels=3).to(DEVICE)
    clients = []
    for i, indices in enumerate(client_indices):
        if len(indices) >= 30:
            subset = Subset(train_data, indices)
            clients.append(FLClient(i, subset, copy.deepcopy(base_model), DEVICE))
        else:
            clients.append(None)
    
    active_clients = [c for c in clients if c is not None and c.has_data]
    num_valid_clients = len(active_clients)
    actual_malicious = min(NUM_MALICIOUS, max(2, num_valid_clients // 5))
    
    global_model = copy.deepcopy(base_model)
    global_weights = global_model.state_dict()
    
    for round_idx in range(NUM_ROUNDS):
        client_weights = []
        for i, client in enumerate(clients):
            if client is None or not client.has_data:
                continue
            client.set_weights(copy.deepcopy(global_weights))
            honest_weights = client.train(local_epochs=LOCAL_EPOCHS)
            if honest_weights is None:
                continue
            is_malicious = (i < actual_malicious)
            if is_malicious:
                from attacks.baruch_attack import baruch_attack
                poisoned = baruch_attack(global_weights, honest_weights, ATTACK_STRENGTH)
                client_weights.append(poisoned)
            else:
                client_weights.append(honest_weights)
        
        if len(client_weights) == 0:
            continue
        
        global_weights = defense_func(client_weights, actual_malicious) if defense_name == "Krum" else defense_func(client_weights)
        
        if global_weights is not None:
            global_model.load_state_dict(global_weights)
            acc = evaluate_model(global_model, test_loader, DEVICE)
            if (round_idx + 1) % 10 == 0:
                print(f"    Round {round_idx + 1}: Acc = {acc:.4f}")
    
    final_acc = evaluate_model(global_model, test_loader, DEVICE)
    print(f"    Final accuracy: {final_acc:.4f}")
    return final_acc

def main():
    print("=" * 70)
    print("COMPLETING DEFENSE EXPERIMENTS")
    print(f"Attack Strength: {ATTACK_STRENGTH}, Rounds: {NUM_ROUNDS}")
    print("=" * 70)
    
    # FedAvg (already have result: 0.0713)
    print("\n✓ FedAvg (No Defense): 0.0713 (from previous run)")
    
    # Run Trimmed Mean
    trimmed_acc = run_defense_experiment("Trimmed Mean", trimmed_mean_aggregate)
    
    # Run Krum
    krum_acc = run_defense_experiment("Krum", lambda w, f=4: krum_aggregate(w, f))
    
    # Final summary
    print("\n" + "=" * 70)
    print("DEFENSE COMPARISON RESULTS (Attack z=3.0)")
    print("=" * 70)
    print(f"FedAvg (No Defense):  0.0713")
    print(f"Trimmed Mean:         {trimmed_acc:.4f}")
    print(f"Krum:                 {krum_acc:.4f}")
    
    if trimmed_acc > 0.5:
        print("\n✅ Trimmed Mean successfully defended against the attack!")
    else:
        print("\n⚠️ Trimmed Mean had difficulty - attack may be too strong")
    
    if krum_acc > 0.5:
        print("✅ Krum successfully defended against the attack!")
    else:
        print("⚠️ Krum had difficulty - attack may be too strong")
    
    print("\n✅ Defense experiments complete!")

if __name__ == "__main__":
    main()
