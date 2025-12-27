    
import numpy as np
from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = {}
        self.v_weights = {}
        self.m_biases = {}
        self.v_biases = {}
        self.t = 0

    def update(self, layer):
        if hasattr(layer, 'weights'):
            if layer not in self.m_weights:
                self.m_weights[layer] = np.zeros_like(layer.weights)
                self.v_weights[layer] = np.zeros_like(layer.weights)
                self.m_biases[layer] = np.zeros_like(layer.biases)
                self.v_biases[layer] = np.zeros_like(layer.biases)

            self.t += 1

            # Update weights
            self.m_weights[layer] = self.beta1 * self.m_weights[layer] + (1 - self.beta1) * layer.weights_grad
            self.v_weights[layer] = self.beta2 * self.v_weights[layer] + (1 - self.beta2) * (layer.weights_grad ** 2)

            m_hat_weights = self.m_weights[layer] / (1 - self.beta1 ** self.t)
            v_hat_weights = self.v_weights[layer] / (1 - self.beta2 ** self.t)

            layer.weights -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)

            # Update biases
            self.m_biases[layer] = self.beta1 * self.m_biases[layer] + (1 - self.beta1) * layer.biases_grad
            self.v_biases[layer] = self.beta2 * self.v_biases[layer] + (1 - self.beta2) * (layer.biases_grad ** 2)

            m_hat_biases = self.m_biases[layer] / (1 - self.beta1 ** self.t)
            v_hat_biases = self.v_biases[layer] / (1 - self.beta2 ** self.t)

            layer.biases -= self.learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)

