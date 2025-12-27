from .network import NeuralNetwork
from .trainer import Trainer

from .layers.dense import Dense
from .layers.activations import Sigmoid, ReLU, Tanh, Softmax
from .layers.batch_norm import BatchNormalization
from .layers.dropout import Dropout

from .optimizers.sgd import SGD
from .optimizers.momentum import Momentum
from .optimizers.adam import Adam   
from .optimizers.adagrad import AdaGrad

from .losses.mse import MSE
from .losses.softmax_ce import SoftmaxCrossEntropy

class HyperparameterTuning:
    '''
    كلاس بيستخدم للبحث عن أفضل هايبر باراميترز (مثل learning rate و batch size)
    عن طريق تجربة مجموعات مختلفة منهم وتقييم أداء الموديل على (validation data)
    اختيار أفضل بارامترات بكون بناءً على أعلى دقة وأدنى خسارة
    '''
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def find_best_params(self, model_builder, opt_class, loss_class, lrs, batch_sizes, epochs=100):
        best_acc = -1.0
        best_loss = float('inf')
        best_config = {}

        for lr in lrs:
            for batch_size in batch_sizes:
                print(f"\nTesting:  LR={lr} ,  Batch={batch_size}")
                
                model = model_builder() 
                optimizer = opt_class(learning_rate=lr) 
                loss_fn = loss_class() 
                trainer = Trainer(model, optimizer, loss_fn) 
                
                trainer.fit(self.x_train, self.y_train, self.x_val, self.y_val, epochs=epochs, batch_size=batch_size)
                
                current_loss = trainer.calculate_loss(self.x_val, self.y_val)
                current_acc = trainer.calculate_accuracy(self.x_val, self.y_val)
                
                if (current_acc > best_acc) or (current_acc == best_acc and current_loss < best_loss):
                    best_acc = current_acc
                    best_loss = current_loss
                    best_config = {
                        'lr': lr, 
                        'batch_size': batch_size,
                        'acc': current_acc
                    }

        print(f"\n === Final Best Configuration === ")
        print(f"Config: LR={best_config['lr']}, Batch={best_config['batch_size']}, With accuracy = {best_acc:.2f}%")
        
        return best_config