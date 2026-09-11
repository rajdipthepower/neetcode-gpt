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
        z1 = np.dot(W1,x)+b1 # W1 @ x + b1 input to hidden layer
        a1 = np.maximum(0.0,z1) # Relu activation on output z1
        predictions = np.dot(W2,a1)+b2 # Hidden to Output Layer 
        loss = np.mean((predictions-y_true)**2) # Mean squared loss 
        error = (2.0/len(predictions))*(predictions-y_true) 
        # dLoss/dpredictions shape:(neurons in output layer,)
        dW2 = np.outer(error,a1) 
        # gradient of Loss wrt W2 Shape: (neurons in output layer,neurons in hidden layer)
        db2 = error # d(Loss)/d(bias) Shape:(numner of neurons in output,)
        dL_da1 = np.transpose(W2) @ error 
        # d(Loss)/d(activations) Shape:(no of neurons in hidden layer,)
        dL_dz1 = dL_da1 * (a1>0) # d(Loss)/d(pre_activation) Shape:(no of neurons in hidden layer,)
        # a1>0 is basically the derivative of ReLU function creating a mask of 1 and 0
        dW1 = np.outer(dL_dz1,x) 
        # # gradient of Loss wrt W1 Shape: (neurons in hidden layer,no of inputs) 
        db1 = dL_dz1 # d(Loss)/d(bias) Shape:(numner of neurons in hidden layer,)
        return {'loss':round(loss,5),'dW1':np.round(dW1,4).tolist(),'db1':np.round(db1,4).tolist(),'dW2':np.round(dW2,4).tolist(),'db2':np.round(db2,4).tolist()}
