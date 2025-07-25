import torch
import torch.nn as nn
import math

class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.is_first = is_first
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1 / self.linear.in_features
            else:
                bound = math.sqrt(6 / self.linear.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class PINN_ID(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=6, output_dim=2, omega_0=30.0, omega_1=1.0):
        """
        SIREN-based PINN model for complex-valued wavefunction learning.

        Assumes input xyt is normalized to [-1, 1]^3.
        """
        super().__init__()
        layers = [SIRENLayer(input_dim, hidden_dim, is_first=True, omega_0=omega_0)]
        for _ in range(num_layers - 2):
            layers.append(SIRENLayer(hidden_dim, hidden_dim, omega_0=omega_1))
        layers.append(nn.Linear(hidden_dim, output_dim))  # final layer: linear output
        self.net = nn.Sequential(*layers)

    def forward(self, xyt):
        out = self.net(xyt)
        u = out[:, 0:1]
        v = out[:, 1:2]
        return u, v
