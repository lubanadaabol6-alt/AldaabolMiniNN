import numpy as np
from .layer import Layer

class Dropout(Layer):
    '''
    كلاس بيطبق تقنية الإسقاط (Dropout) على الطبقة في الشبكة العصبية
    خلال التدريب، بيتم إسقاط بعض الوحدات بشكل عشوائي بناءً على
    احتمالية الإسقاط المحددة، مما يساعد في تقليل الـ (overfitting)
    وهيك بعزز من قدرة النموذج على التعميم
    '''
    def __init__(self, drop_probability):
        super().__init__("Dropout")
        self.drop_probability = drop_probability
        self.mask = None

    def forward(self, input_data, training=True):
        if training:
            self.mask = (np.random.rand(*input_data.shape) > self.drop_probability) / (1.0 - self.drop_probability)
            return input_data * self.mask
        else:
            return input_data

    def backward(self, output_gradient):
        return output_gradient * self.mask