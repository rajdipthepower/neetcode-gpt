import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        return np.round(1/(1+np.exp(-z)), 5) 
    # Time complexity: O(n) and Space Complexity : O(n) since we store 1/(1+np.exp(-z) before applying np.round()

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        return np.maximum(0,z) # O(1) space complexity ,O(n):Time complexity
        # return np.array([max(0,i) for i in z],dtype = np.float64) -> O(n) space complexity for storing the list before applying np.array()
