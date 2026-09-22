import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        record = []
        with torch.no_grad():
            for module in model.modules():
                s = {} # dictionary initialized within loop to allocate each layer to separate dictionaries or else they become repeated copies
                if isinstance(module,nn.Sequential):
                    del s
                    continue
                out = module(x)
                if isinstance(module,nn.Linear):
                    summed_activations = torch.sum((out<=0),axis = 0)
                    s['mean'] = round(torch.mean(out).item(),4)
                    s['std'] = round(torch.std(out).item(),4)
                    s['dead_fraction'] = round((summed_activations == out.shape[0]).sum().item()/out.shape[1],4)
                    record.append(s)
                x = out
                del s
            return record


    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        record = []
        model.zero_grad()
        loss = nn.MSELoss()(model(x),y)
        loss.backward()

        for module in model.modules():
            s = {}
            if isinstance(module,nn.Linear):
                s['mean'] = round(torch.mean(module.weight.grad).item(),4)
                s['std'] = round(torch.std(module.weight.grad).item(),4)
                s['norm'] = round(torch.linalg.vector_norm(module.weight.grad,ord = 2).item(),4)
                record.append(s)
            del s
        return record

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for i in activation_stats:
            if i['dead_fraction'] > 0.5:
                return 'dead_neurons'
        for j,i in zip(gradient_stats,activation_stats):
            if j['norm'] > 1000:
                return 'exploding_gradients'
            elif j['norm'] < 1e-5:
                return 'vanishing_gradients'
            elif i['std'] <0.1:
                return 'vanishing_gradients'
            elif i['std'] > 10.0:
                return 'exploding_gradients'
        return 'healthy'
             
