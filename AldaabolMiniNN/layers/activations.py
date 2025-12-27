import numpy as np
from .layer import Layer

# Activation Functions
class ReLU(Layer):
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, output_gradient):
        return output_gradient * (self.input > 0)

class Sigmoid(Layer):
    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, output_gradient):
        return output_gradient * (self.output * (1 - self.output))
    
class Tanh(Layer):
    def forward(self, input_data):
        self.output = np.tanh(input_data)
        return self.output

    def backward(self, output_gradient):
        return output_gradient * (1 - self.output ** 2)
    
class Softmax(Layer):
    def forward(self, input_data):
        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient):
        # Note: Softmax backward is usually combined with Cross-Entropy loss for efficiency
        return output_gradient  