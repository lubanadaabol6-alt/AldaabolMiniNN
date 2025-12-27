import numpy as np

class SoftmaxCrossEntropy:
    '''
    هاد الكلاس بيحسب خسارة الـ Softmax مع Cross-Entropy بين القيم الحقيقية والقيم المتوقعة.
    '''
    def loss(self, y_true, y_pred):
        exps = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        self.softmax = exps / np.sum(exps, axis=1, keepdims=True)
        
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(self.softmax + 1e-9)) / m
        return loss

    def backward(self, y_true, y_pred):
        m = y_true.shape[0]
        return (self.softmax - y_true) / m

