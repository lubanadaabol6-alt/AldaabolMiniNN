
    
from .optimizer import Optimizer

import numpy as np  

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities_weights = {}
        self.velocities_biases = {}

    def update(self, layer):
        if hasattr(layer, 'weights'):
            if layer not in self.velocities_weights:
                self.velocities_weights[layer] = np.zeros_like(layer.weights)
                self.velocities_biases[layer] = np.zeros_like(layer.biases)

            # Update weights
            self.velocities_weights[layer] = (self.momentum * self.velocities_weights[layer] - 
                                              self.learning_rate * layer.weights_grad)
            layer.weights += self.velocities_weights[layer]

            # Update biases
            self.velocities_biases[layer] = (self.momentum * self.velocities_biases[layer] - 
                                             self.learning_rate * layer.biases_grad)
            layer.biases += self.velocities_biases[layer]
