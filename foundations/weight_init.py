import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        weights = torch.randn(fan_out,fan_in)*math.sqrt(2/(fan_in + fan_out))
        return torch.round(weights,decimals = 4,out=weights).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        weights = torch.randn(fan_out,fan_in)*math.sqrt(2/fan_in)
        return torch.round(weights,decimals = 4,out=weights).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        std_layer = []
        rng = torch.get_rng_state()
        for i in range(num_layers):
            _ = torch.randn(hidden_dim,input_dim if i==0 else hidden_dim)
        x = torch.randn(input_dim)
        a = x
        torch.set_rng_state(rng)
        for i in range(num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2 / (len(a) + hidden_dim))
            elif init_type == 'kaiming':
                std = math.sqrt(2 / len(a))
            else:
                std = 1.0
            weight = torch.randn(hidden_dim,len(a)) * std
            a = torch.relu(a @ weight.T)
            std_layer.append(round(torch.std(a).item(),2))
        return std_layer