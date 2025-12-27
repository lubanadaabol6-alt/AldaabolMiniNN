import numpy as np

class MSE:
    '''     
    هاد الكلاس بيحسب خسارة متوسط مربع الخطأ (Mean Squared Error) بين القيم الحقيقية والقيم المتوقعة.
    '''
    @staticmethod
    def loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def backward(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size