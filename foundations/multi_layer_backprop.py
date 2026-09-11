import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        z1 = np.dot(W1,x)+b1
        a1 = np.maximum(0.0,z1)
        predictions = np.dot(W2,a1)+b2
        loss = np.mean((predictions-y_true)**2)
        error = (2.0/len(predictions))*(predictions-y_true)
        dW2 = np.outer(error,a1)
        db2 = error
        dL_da1 = np.transpose(W2) @ error
        dL_dz1 = dL_da1 * (a1>0)
        dW1 = np.outer(dL_dz1,x)
        db1 = dL_dz1
        return {'loss':round(loss,5),'dW1':np.round(dW1,4).tolist(),'db1':np.round(db1,4).tolist(),'dW2':np.round(dW2,4).tolist(),'db2':np.round(db2,4).tolist()}
