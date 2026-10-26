"""
CNN model for BloodMNIST (3 channels, 8 classes).

Uses GroupNorm rather than BatchNorm: GroupNorm keeps no running statistics,
so the model state is exactly its trainable parameters. This avoids averaging
(or attacking) non-trainable buffers across non-IID clients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MedicalCNN(nn.Module):
    def __init__(self, num_classes=8, input_channels=3, num_groups=8):
        super(MedicalCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_model(num_classes=8, input_channels=3, num_groups=8):
    return MedicalCNN(num_classes, input_channels, num_groups)
