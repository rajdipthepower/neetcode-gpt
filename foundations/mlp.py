import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        a_i = x
        buffer_memory = np.empty(max(i.shape[1] for i in weights))
        for i in range(len(weights)):
            buffer_pointer = buffer_memory[:weights[i].shape[1]]
            z = np.dot(a_i,weights[i])
            buffer_pointer[:] = z
            buffer_pointer += biases[i]
            if i == len(weights) - 1:
                a_i = buffer_pointer
                break
            np.maximum(0.0,buffer_pointer,out = buffer_pointer)
            a_i = buffer_pointer

        return np.round(a_i,5,out=a_i)

