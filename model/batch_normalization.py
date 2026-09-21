import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        if training:
            mu = np.mean(x,axis = 0)
            var = np.var(x,axis = 0)
            x_hat = (x - mu)/np.sqrt(var + eps)
            for i in range(len(running_mean)):
                running_mean[i] = (1-momentum) * running_mean[i] + momentum * mu[i]
                running_var[i] = (1-momentum) * running_var[i] + momentum * var[i]
            return (np.round(gamma * x_hat + beta,4),np.round(running_mean,4),np.round(running_var,4))
        else:
            x_hat = (x - np.array(running_mean))/np.sqrt(np.array(running_var) + eps)
            return (np.round(gamma * x_hat + beta,4),np.round(running_mean,4),np.round(running_var,4))
