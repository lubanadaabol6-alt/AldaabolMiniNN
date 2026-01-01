from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from AldaabolMiniNN import (
    NeuralNetwork,Trainer, HyperparameterTuning,
    Dense, Sigmoid, ReLU, Tanh, BatchNormalization, Dropout,
    SGD, Adam, Momentum, AdaGrad,
    MSE, SoftmaxCrossEntropy, 
)


# load and prepare data
iris = load_iris()
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(iris.target.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(iris.data, y_onehot, test_size=0.2, random_state=42)

# built the test model
def build_model():
    model = NeuralNetwork()
    model.add(Dense(4, 32, init = 'he'))     
    model.add(Sigmoid())
    model.add(BatchNormalization(32))
    model.add(Dense(32, 16, init = 'he'))
    model.add(ReLU())
    model.add(Dense(16, 3, init = 'xaviar'))      
    return model

# search for best hyperparameters
tuning = HyperparameterTuning(X_train, y_train, X_test, y_test)

best_params = tuning.find_best_params(
    model_builder = build_model, 
    opt_class     = SGD,                
    loss_class    = SoftmaxCrossEntropy, 
    lrs           = [0.1, 0.01, 0.001], 
    batch_sizes   = [8, 16, 32], 
    epochs        = 300
)

# train final model with best hyperparameters
final_optimizer = SGD(learning_rate=best_params['lr'])
final_loss      = SoftmaxCrossEntropy()
final_model     = build_model()

trainer = Trainer(final_model, final_optimizer, final_loss)
trainer.fit(X_train, y_train, X_test, y_test, epochs=1000, batch_size=best_params['batch_size'])