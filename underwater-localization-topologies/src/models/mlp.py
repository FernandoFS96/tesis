# src/models/mlp.py
"""MLP Model Architecture."""
import torch
import torch.nn as nn
import torch.nn.init as init

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, neurons):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, output_dim)
        self.relu = nn.ReLU()

        # init
        for m in [self.fc1, self.fc2]:
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)