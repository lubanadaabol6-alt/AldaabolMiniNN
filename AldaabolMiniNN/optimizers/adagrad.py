
from .optimizer import Optimizer
import numpy as np

class AdaGrad(Optimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.sum_squared_weights = {}
        self.sum_squared_biases = {}

    def update(self, layer):
        if hasattr(layer, 'weights'):
            if layer not in self.sum_squared_weights:
                self.sum_squared_weights[layer] = np.zeros_like(layer.weights)
                self.sum_squared_biases[layer] = np.zeros_like(layer.biases)

            # Update weights
            self.sum_squared_weights[layer] += layer.weights_grad ** 2
            adjusted_lr_weights = self.learning_rate / (np.sqrt(self.sum_squared_weights[layer]) + self.epsilon)
            layer.weights -= adjusted_lr_weights * layer.weights_grad

            # Update biases
            self.sum_squared_biases[layer] += layer.biases_grad ** 2
            adjusted_lr_biases = self.learning_rate / (np.sqrt(self.sum_squared_biases[layer]) + self.epsilon)
            layer.biases -= adjusted_lr_biases * layer.biases_grad