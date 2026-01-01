## Mini Neural Network Library

This project is a **mini neural network library built from scratch using NumPy only**, without using frameworks such as PyTorch or TensorFlow.

---

## Project Features

* Build neural networks using different layers
* Train the network using backpropagation
* Support multiple optimizers
* Use different loss functions
* Perform hyperparameter tuning
* Train and evaluate on real datasets

## Implemented Components

### Neural Network

* Manages layers in sequence
* Performs forward prediction through all layers

---

### Layers

* **Dense (Fully Connected)**
* **Activation Functions:** ReLU, Sigmoid, Tanh
* **Batch Normalization**
* **Dropout**

---

### Loss Functions

* **MSE** (for regression)
* **Softmax Cross-Entropy** (for multi-class classification)

---

### Optimizers

* **SGD**
* **Momentum**
* **Adam**
* **AdaGrad**

---

### Trainer

* Handles the training process
* Performs forward and backward propagation
* Updates weights using the optimizer
* Computes loss and accuracy
* Supports mini-batch training

---

### Hyperparameter Tuning

* Tests different learning rates and batch sizes
* Selects the best parameters based on validation accuracy 

---

## Experiment 1

* Dataset: **Iris Dataset**
* Loss fun: **SoftmaxCrossEnrtopy**
* Optimizer: **SGD**
* Data split into training and validation sets
* Achieved accuracy up to **~100%**


*OUTPUT EXAMPLE (1):*
Testing:  LR=0.1 ,  Batch=8
Epoch 0:   Loss 1.1194,   Validation Accuracy: 73.33%
Epoch 100:   Loss 0.4267,   Validation Accuracy: 90.00%
Epoch 200:   Loss 0.1521,   Validation Accuracy: 86.67%
Epoch 300:   Loss 0.1873,   Validation Accuracy: 96.67%

Testing:  LR=0.1 ,  Batch=16
Epoch 0:   Loss 1.0977,   Validation Accuracy: 63.33%
Epoch 100:   Loss 0.2217,   Validation Accuracy: 96.67%
Epoch 200:   Loss 0.1970,   Validation Accuracy: 100.00%
Epoch 300:   Loss 0.1392,   Validation Accuracy: 100.00%

Testing:  LR=0.1 ,  Batch=32
Epoch 0:   Loss 1.0977,   Validation Accuracy: 56.67%
Epoch 100:   Loss 0.3759,   Validation Accuracy: 100.00%
Epoch 200:   Loss 0.1703,   Validation Accuracy: 100.00%
Epoch 300:   Loss 0.0587,   Validation Accuracy: 93.33%

Testing:  LR=0.01 ,  Batch=8
Epoch 0:   Loss 1.0973,   Validation Accuracy: 30.00%
Epoch 100:   Loss 1.0073,   Validation Accuracy: 63.33%
Epoch 200:   Loss 0.7071,   Validation Accuracy: 80.00%
Epoch 300:   Loss 0.4752,   Validation Accuracy: 80.00%

Testing:  LR=0.01 ,  Batch=16
Epoch 0:   Loss 1.0992,   Validation Accuracy: 53.33%
Epoch 100:   Loss 1.0192,   Validation Accuracy: 70.00%
Epoch 200:   Loss 0.8490,   Validation Accuracy: 70.00%
Epoch 300:   Loss 0.6343,   Validation Accuracy: 76.67%

Testing:  LR=0.01 ,  Batch=32
Epoch 0:   Loss 1.0992,   Validation Accuracy: 16.67%
Epoch 100:   Loss 1.0466,   Validation Accuracy: 80.00%
Epoch 200:   Loss 0.9879,   Validation Accuracy: 80.00%
Epoch 300:   Loss 0.9469,   Validation Accuracy: 73.33%

Testing:  LR=0.001 ,  Batch=8
Epoch 0:   Loss 1.0916,   Validation Accuracy: 23.33%
Epoch 100:   Loss 1.0538,   Validation Accuracy: 93.33%
Epoch 200:   Loss 1.0364,   Validation Accuracy: 93.33%
Epoch 300:   Loss 1.0124,   Validation Accuracy: 96.67%

Testing:  LR=0.001 ,  Batch=16
Epoch 0:   Loss 1.0997,   Validation Accuracy: 36.67%
Epoch 100:   Loss 1.0517,   Validation Accuracy: 70.00%
Epoch 200:   Loss 1.0713,   Validation Accuracy: 70.00%
Epoch 300:   Loss 1.0394,   Validation Accuracy: 70.00%

Testing:  LR=0.001 ,  Batch=32
Epoch 0:   Loss 1.1104,   Validation Accuracy: 0.00%
Epoch 100:   Loss 1.0928,   Validation Accuracy: 56.67%
Epoch 200:   Loss 1.0954,   Validation Accuracy: 60.00%
Epoch 300:   Loss 1.0901,   Validation Accuracy: 63.33%

 === Final Best Configuration ===
Config: LR=0.1, Batch=16, With accuracy = 100.00%
Epoch 0:   Loss 1.0810,   Validation Accuracy: 70.00%
Epoch 100:   Loss 0.2651,   Validation Accuracy: 96.67%
Epoch 200:   Loss 0.3080,   Validation Accuracy: 100.00%
Epoch 300:   Loss 0.3404,   Validation Accuracy: 86.67%
Epoch 400:   Loss 0.1250,   Validation Accuracy: 96.67%
Epoch 500:   Loss 0.0358,   Validation Accuracy: 100.00%
Epoch 600:   Loss 0.0386,   Validation Accuracy: 100.00%
Epoch 700:   Loss 0.0635,   Validation Accuracy: 90.00%
Epoch 800:   Loss 0.2695,   Validation Accuracy: 96.67%
Epoch 900:   Loss 0.0363,   Validation Accuracy: 100.00%
Epoch 1000:   Loss 0.0124,   Validation Accuracy: 93.33%


## Experiment 2

* Dataset: **Digits Dataset**
* Loss fun: **SoftmaxCrossEnrtopy**
* Optimizer: **AbaGrad**
* Achieved accuracy up to **~99%**

*OUTPUT EXAMPLE (2):*
Testing:  LR=0.1 ,  Batch=8
Epoch 0:   Loss 0.3851,   Validation Accuracy: 93.33%
Epoch 100:   Loss 0.0008,   Validation Accuracy: 98.61%
Epoch 200:   Loss 0.0000,   Validation Accuracy: 99.17%
Epoch 300:   Loss 0.0000,   Validation Accuracy: 98.89%

Testing:  LR=0.1 ,  Batch=16
Epoch 0:   Loss 0.3540,   Validation Accuracy: 92.50%
Epoch 100:   Loss 0.0018,   Validation Accuracy: 98.61%
Epoch 200:   Loss 0.0057,   Validation Accuracy: 97.78%
Epoch 300:   Loss 0.0004,   Validation Accuracy: 98.06%

Testing:  LR=0.1 ,  Batch=32
Epoch 0:   Loss 0.8228,   Validation Accuracy: 84.72%
Epoch 100:   Loss 0.0048,   Validation Accuracy: 97.50%
Epoch 200:   Loss 0.0018,   Validation Accuracy: 97.78%
Epoch 300:   Loss 0.0007,   Validation Accuracy: 98.06%

Testing:  LR=0.01 ,  Batch=8
Epoch 0:   Loss 2.2865,   Validation Accuracy: 59.44%
Epoch 100:   Loss 0.1369,   Validation Accuracy: 97.78%
Epoch 200:   Loss 0.0163,   Validation Accuracy: 98.61%
Epoch 300:   Loss 0.3243,   Validation Accuracy: 98.33%

Testing:  LR=0.01 ,  Batch=16
Epoch 0:   Loss 1.9945,   Validation Accuracy: 33.61%
Epoch 100:   Loss 0.0504,   Validation Accuracy: 97.78%
Epoch 200:   Loss 0.0651,   Validation Accuracy: 98.06%
Epoch 300:   Loss 0.0037,   Validation Accuracy: 98.61%

Testing:  LR=0.01 ,  Batch=32
Epoch 0:   Loss 2.2226,   Validation Accuracy: 26.39%
Epoch 100:   Loss 0.0388,   Validation Accuracy: 98.06%
Epoch 200:   Loss 0.0261,   Validation Accuracy: 98.33%
Epoch 300:   Loss 0.0215,   Validation Accuracy: 98.33%

Testing:  LR=0.001 ,  Batch=8
Epoch 0:   Loss 2.4079,   Validation Accuracy: 18.33%
Epoch 100:   Loss 0.4756,   Validation Accuracy: 96.39%
Epoch 200:   Loss 0.2877,   Validation Accuracy: 97.22%
Epoch 300:   Loss 1.0002,   Validation Accuracy: 98.06%

Testing:  LR=0.001 ,  Batch=16
Epoch 0:   Loss 2.2172,   Validation Accuracy: 4.44%
Epoch 100:   Loss 0.3459,   Validation Accuracy: 93.06%
Epoch 200:   Loss 0.3620,   Validation Accuracy: 95.56%
Epoch 300:   Loss 0.1275,   Validation Accuracy: 96.39%

Testing:  LR=0.001 ,  Batch=32
Epoch 0:   Loss 2.4066,   Validation Accuracy: 12.78%
Epoch 100:   Loss 0.9555,   Validation Accuracy: 83.06%
Epoch 200:   Loss 0.4447,   Validation Accuracy: 91.39%
Epoch 300:   Loss 0.2497,   Validation Accuracy: 94.72%

 === Final Best Configuration ===
Config: LR=0.1, Batch=8, With accuracy = 98.89%
Epoch 0:   Loss 0.5915,   Validation Accuracy: 96.11%
Epoch 100:   Loss 0.0009,   Validation Accuracy: 98.33%
Epoch 200:   Loss 0.0001,   Validation Accuracy: 98.89%
Epoch 300:   Loss 0.0795,   Validation Accuracy: 98.61%
Epoch 400:   Loss 0.0001,   Validation Accuracy: 98.89%
Epoch 500:   Loss 0.0010,   Validation Accuracy: 98.89%
Epoch 600:   Loss 0.0035,   Validation Accuracy: 98.61%
Epoch 700:   Loss 0.0016,   Validation Accuracy: 98.89%
Epoch 800:   Loss 0.0002,   Validation Accuracy: 99.17%
Epoch 900:   Loss 0.0007,   Validation Accuracy: 98.89%
Epoch 1000:   Loss 0.0000,   Validation Accuracy: 98.89%
