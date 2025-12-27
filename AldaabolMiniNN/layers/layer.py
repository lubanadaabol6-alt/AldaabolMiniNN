class Layer:
    '''
    الكلاس الأساسي لكل الطبقات في الشبكة العصبية
    يحتوي على التوابع الأساسية التي يجب أن تنفذها كل طبقة
    كل طبقة بتم تعريفها بترث من هاد الكلاس
    بحيث بيفرض عليها تنفيذ التوابع forward و backward
    '''
    def __init__(self, name=None):
        self.name = name

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError
