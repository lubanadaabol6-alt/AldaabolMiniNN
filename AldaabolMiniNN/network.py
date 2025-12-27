class NeuralNetwork:
    """
    كلاس بمثل الشبكة العصبية
    بيحتوي مجموعة من الطبقات layers 
    وبيوفر تابع للتنبؤ وتابع التدريب
    """
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train_step(self, x_batch, y_batch, loss_fn, optimizer):
        y_pred = self.predict(x_batch)
        loss = loss_fn.loss(y_batch, y_pred)
        
        grad = loss_fn.backward(y_batch, y_pred)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            optimizer.update(layer)
        return loss