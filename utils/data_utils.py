"""
Data loading and client utilities
"""

import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from medmnist import BloodMNIST, INFO

# Configuration
DATA_ROOT = 'C:/Users/ASUS/.medmnist'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32

class TransformedBloodMNIST(BloodMNIST):
    """BloodMNIST with proper tensor transforms"""
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
        
        # Convert HWC to CHW
        if img.dim() == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        
        img = img / 255.0  # Normalize to [0, 1]
        return img, label

class FLClient:
    """Federated Learning client with local training"""
    def __init__(self, client_id, dataset, model, device, batch_size=32, local_epochs=2):
        self.client_id = client_id
        self.has_data = len(dataset) > 0
        self.device = device
        self.local_epochs = local_epochs
        
        if not self.has_data:
            return
            
        self.model = copy.deepcopy(model).to(device)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
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
    
    def train(self):
        """Perform local training and return updated weights"""
        if not self.has_data:
            return None
            
        self.model.train()
        for epoch in range(self.local_epochs):
            for images, labels in self.dataloader:
                images = images.to(self.device)
                labels = labels.squeeze().long().to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        
        return self.get_weights()

def load_data():
    """Load BloodMNIST dataset"""
    print("Loading BloodMNIST dataset...")
    train_data = TransformedBloodMNIST(split='train', download=True, size=28)
    test_data = TransformedBloodMNIST(split='test', download=True, size=28)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    num_classes = len(INFO['bloodmnist']['label'])
    print(f"  Training samples: {len(train_data)}, Classes: {num_classes}")
    return train_data, test_loader, num_classes

def create_non_iid_splits(dataset, num_clients, alpha=0.5, min_samples=30):
    """
    Create non-IID client splits using Dirichlet distribution.
    
    Args:
        dataset: Training dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (smaller = more non-IID)
        min_samples: Minimum samples per client
    """
    print("  Creating non-IID client splits...")
    
    # Extract labels
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    labels = np.array(labels)
    
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    # Try different alpha values until we meet min_samples requirement
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
            print(f"  Split successful (min samples: {min_samples_actual})")
            break
        current_alpha = min(current_alpha * 1.3, 10.0)
    
    # Print distribution
    for i in range(min(5, num_clients)):
        print(f"  Client {i}: {len(client_indices[i])} samples")
    
    return client_indices