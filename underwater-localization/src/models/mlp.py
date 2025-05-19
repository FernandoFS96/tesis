# src/models/mlp.py
"""MLP Model Architecture."""
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, neurons=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x