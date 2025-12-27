from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer):
        if hasattr(layer, 'weights'):
            layer.weights -= self.learning_rate * layer.weights_grad
            layer.biases -= self.learning_rate * layer.biases_grad