import numpy as np
class Trainer:
    '''
    هاد الكلاس بيشرف على عملية تدريب الشبكة العصبية.
    بيستخدم الـ Optimizer لتحديث الأوزان وبيحسب الخسارة باستخدام دالة الخسارة المحددة.
    '''
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, x_batch, y_batch):
        # forward pass
        y_pred = self.model.predict(x_batch)
        
        # حساب الخسارة
        loss = self.loss_fn.loss(y_batch, y_pred)
        
        # backward pass
        grad = self.loss_fn.backward(y_batch, y_pred)
        for layer in reversed(self.model.layers):
            grad = layer.backward(grad)
            # تحديث الأوزان باستخدام الـ Optimizer 
            self.optimizer.update(layer)
            
        return loss

    def fit(self, x_train, y_train, x_val, y_val, epochs, batch_size=32):
        for epoch in range(epochs + 1):
            # تقسيم البيانات إلى Batches 
            indices = np.random.permutation(len(x_train))
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            for i in range(0, len(x_train), batch_size):
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                loss = self.train_step(x_batch, y_batch)
            
            if epoch % 100 == 0:
                acc = self.calculate_accuracy(x_val, y_val)
                print(f"Epoch {epoch}:   Loss {loss:.4f},   Validation Accuracy: {acc:.2f}%")

    def calculate_accuracy(self, x, y):
        predictions = self.model.predict(x)
        y_pred_labels = np.argmax(predictions, axis=1)
        y_true_labels = np.argmax(y, axis=1)
        return np.mean(y_pred_labels == y_true_labels) * 100
    
    def calculate_loss(self, x, y):
        predictions = self.model.predict(x)
        return self.loss_fn.loss(predictions, y)