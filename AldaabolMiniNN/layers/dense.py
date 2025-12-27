import numpy as np
from .layer import Layer

class Dense(Layer):
    '''
    هاد الكلاس بيمثل طبقة Dense (Fully Connected) في الشبكة العصبية.
    بيحتوي على أوزان وانحيازات، وبيوفر توابع للـ (forward pass)
    والـ (backward pass)
    لحساب التدرجات اللازمة لتحديث الأوزان خلال التدريب.
    '''
    def __init__(self, input_size, output_size):
        super().__init__("Dense")
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2./input_size)
        self.biases = np.zeros((1, output_size))
        self.weights_grad = None
        self.biases_grad = None

    def forward(self, input_data):
        self.input = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, output_gradient):
        self.weights_grad = np.dot(self.input.T, output_gradient)
        self.biases_grad = np.sum(output_gradient, axis=0, keepdims=True)
        return np.dot(output_gradient, self.weights.T)