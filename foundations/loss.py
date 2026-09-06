import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred = np.clip(y_pred,1e-7, 1 - 1e-7)
        result = -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
        return np.round(result,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred = y_pred[np.arange(len(y_pred)),np.argmax(y_true,axis=1)]
        res = -np.mean(np.log(np.clip(y_pred,1e-7, 1 - 1e-7))) 
        # Applying np.clip() on 1D array , so space complexity:O(N)
        # Time complexity:O(N*C) due to np.argmax() for every row,all columns C
        return np.round(res,4)
