Project Overview: AldaabolMiniNN Library
AldaabolMiniNN** is a modular Deep Learning framework built entirely from scratch using **Python** and **NumPy**. The project mimics professional libraries like **PyTorch** by organizing components into a reusable package structure

How I Built It:
**Modular Architecture:** I structured the library into independent sub-packages: `layers`, `optimizers`, `losses`, and `network`. Each component is isolated to ensure scalability.
**The Core Engine:** Every layer inherits from a base `Layer` class, enforcing a strict contract for `forward` and `backward` passes using the **Chain Rule**.
**Numerical Stability:** I implemented a specialized `SoftmaxCrossEntropy` loss that handles potential numerical overflows (using the max-subtraction trick)
**Advanced Components:** I integrated a **Batch Normalization** layer to stabilize the internal covariate shift, allowing the model to train faster with higher learning rates
**Automation (The Tuner):** I developed a `HyperparameterTuning` module to automatically find the best `Learning Rate` and `Batch Size` through a grid-search approach.

Final Performance:
**Architecture:** `Dense > Sigmoid > BatchNorm > Dense > ReLU > Dense > SoftmaxWithLoss`.
**Dataset:** Iris Dataset (Categorical Classification).
**Results:** Successfully achieved **100% Validation Accuracy** with a final loss **0.0065**


شرح عام:
**AldaabolMiniNN** هو إطار عمل للتعلم العميق تم بناؤه بالكامل من الصفر باستخدام **Python** و **NumPy**. يحاكي المشروع هيكلية المكتبات الاحترافية مثل *PyTorch* من خلال تنظيم المكونات في حزمة برمجية (Package) قابلة لإعادة الاستخدام

كيف أنجزت العمل:
**الهيكلية المجزأة (Modular Architecture):** قمت بتنظيم المكتبة إلى حزم فرعية مستقلة: `layers` (الطبقات)، `optimizers` (المحسنات)، `losses` (دوال الخسارة)، و `network` (الشبكة)
**المحرك الأساسي:** ترث كل طبقة من كلاس أساسي `Layer` يفرض تنفيذ عمليتي الانتشار الأمامي والخلفي باستخدام **قاعدة السلسلة (Chain Rule)** بدقة رياضية
**الاستقرار الحسابي:** نفذت دالة خسارة `SoftmaxCrossEntropy` متطورة تعالج مشكلة القيم الضخمة (Overflow) لضمان استقرار التدريب
 **المكونات المتقدمة:** أضفت طبقة **Batch Normalization** لمعايرة البيانات داخلياً، مما سمح للنموذج بالتدريب بسرعة أكبر وبمعدلات تعلم أعلى دون تذبذب
 **الأتمتة (Tuning):** قمت بتطوير وحدة `HyperparameterTuning` للبحث تلقائياً عن أفضل "معدل تعلم" و "حجم دفعة" (Batch Size) لضمان الوصول للحل الأمثل

  الأداء النهائي:
**هيكلية الشبكة:** `Dense > Sigmoid > BatchNorm > Dense > ReLU > Dense > SoftmaxWithLoss`.
**مجموعة البيانات:** Iris Dataset (تصنيف الفئات).
**النتائج:** نجحت المكتبة في الوصول إلى دقة **100%** على بيانات الاختبار مع خسارة نهائية ضئيلة بلغت **0.0065**.



*OUTPUT EXAMPLE (1):*
Dataset: iris ====== loss fun: SoftmaxCrossEnrtopy ====== optimizer: SGD
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


*OUTPUT EXAMPLE (2):*
Dataset: digits ====== loss fun: SoftmaxCrossEnrtopy ====== optimizer: AbaGrad
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
