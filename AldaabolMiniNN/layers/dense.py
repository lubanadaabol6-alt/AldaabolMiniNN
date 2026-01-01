import numpy as np
from .layer import Layer

class Dense:
    '''
    كلاس بيمثل طبقة كثيفة (Dense Layer) في الشبكة العصبية
    هي الطبقة بتعمل تحول خطي للمدخلات عن طريق ضربها في مصفوفة الأوزان
    وإضافة الانحياز (biases). 
    بيدعم تهيئة الأوزان باستخدام طرق مختلفة متل "xavier" و "he" أو تهيئة عشوائية بسيطة
    ﻷداء أفضل في التدريب.
    '''
    def __init__(self, in_features, out_features, init="xavier"):
        self.in_features = in_features
        self.out_features = out_features

        if init == "he":
            self.Weights = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        elif init == "xavier":
            self.Weights = np.random.randn(in_features, out_features) * np.sqrt(1.0 / in_features)
        else:
            self.Weights = np.random.randn(in_features, out_features) * 0.01

        self.biases = np.zeros((1, out_features))


    def forward(self, input_data):
        self.input = input_data
        return np.dot(input_data, self.Weights) + self.biases

    def backward(self, output_gradient):
        self.weights_grad = np.dot(self.input.T, output_gradient)
        self.biases_grad = np.sum(output_gradient, axis=0, keepdims=True)
        return np.dot(output_gradient, self.Weights.T)