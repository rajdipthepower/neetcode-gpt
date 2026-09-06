import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        return np.round(weights @ X.T,5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        model_prediction -= ground_truth # No extra memory
        return np.round(np.dot(model_prediction.reshape(-1), model_prediction.reshape(-1))/len(model_prediction),5)
# reshape creates a view and don't allocate memory whereas dot directly evaluates vector dot product without creating temporary array making Space complexity:O(1)