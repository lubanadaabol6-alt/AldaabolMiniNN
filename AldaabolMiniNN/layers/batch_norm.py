import numpy as np
from .layer import Layer

class BatchNormalization(Layer):
    '''
    كلاس بيطبق تقنية التطبيع الداخلي للدفعات (Batch Normalization)
    على الطبقة في الشبكة العصبية. هالتقنية بتساعد في تسريع
    وتحسين استقرار التدريب عن طريق تطبيع مخرجات الطبقة السابقة
    وتقليل التغيرات الداخلية في التوزيع الإحصائي للمدخلات.
    '''
    def __init__(self, epsilon=1e-8):
        super().__init__("BatchNormalization")
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
        self.input_normalized = None
        self.mean = None
        self.variance = None

    def forward(self, input_data):
        self.input = input_data 
        
        if self.gamma is None:
            self.gamma = np.ones((1, input_data.shape[1]))
        if self.beta is None:
            self.beta = np.zeros((1, input_data.shape[1]))

        self.mean = np.mean(input_data, axis=0)
        self.variance = np.var(input_data, axis=0)
        self.input_normalized = (input_data - self.mean) / np.sqrt(self.variance + self.epsilon)
        return self.gamma * self.input_normalized + self.beta

    def backward(self, output_gradient):
        m = output_gradient.shape[0]
        dbeta = np.sum(output_gradient, axis=0)
        dgamma = np.sum(output_gradient * self.input_normalized, axis=0)

        dinput_normalized = output_gradient * self.gamma
        dvariance = np.sum(dinput_normalized * (self.input - self.mean) * -0.5 * (self.variance + self.epsilon) ** -1.5, axis=0)
        dmean = np.sum(dinput_normalized * -1 / np.sqrt(self.variance + self.epsilon), axis=0) + dvariance * np.mean(-2 * (self.input - self.mean), axis=0)

        dinput = dinput_normalized / np.sqrt(self.variance + self.epsilon) + dvariance * 2 * (self.input - self.mean) / m + dmean / m

        self.gamma_grad = dgamma
        self.beta_grad = dbeta
        return dinput