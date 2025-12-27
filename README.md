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



للتشغيل من مجلد البروجكت AldaabolMiniNNproject
py -m test.tester


*OUTPUT EXAMPLE:*
Testing:  LR=0.1 ,  Batch=8
Epoch 0:   Loss 0.3059,   Validation Accuracy: 100.00%
Epoch 100:   Loss 0.0243,   Validation Accuracy: 93.33%
Epoch 200:   Loss 0.0746,   Validation Accuracy: 100.00%
Epoch 300:   Loss 0.0272,   Validation Accuracy: 100.00%

Testing:  LR=0.1 ,  Batch=16
Epoch 0:   Loss 0.3667,   Validation Accuracy: 96.67%
Epoch 100:   Loss 0.0107,   Validation Accuracy: 96.67%
Epoch 200:   Loss 0.2920,   Validation Accuracy: 93.33%
Epoch 300:   Loss 0.0065,   Validation Accuracy: 100.00%

Testing:  LR=0.1 ,  Batch=32
Epoch 0:   Loss 0.6357,   Validation Accuracy: 63.33%
Epoch 100:   Loss 0.0132,   Validation Accuracy: 93.33%
Epoch 200:   Loss 0.0168,   Validation Accuracy: 93.33%
Epoch 300:   Loss 0.1973,   Validation Accuracy: 93.33%

Testing:  LR=0.01 ,  Batch=8
Epoch 0:   Loss 1.1896,   Validation Accuracy: 83.33%
Epoch 100:   Loss 0.1609,   Validation Accuracy: 96.67%
Epoch 200:   Loss 0.0309,   Validation Accuracy: 96.67%
Epoch 300:   Loss 0.1746,   Validation Accuracy: 96.67%

Testing:  LR=0.01 ,  Batch=16
Epoch 0:   Loss 1.2922,   Validation Accuracy: 53.33%
Epoch 100:   Loss 0.0963,   Validation Accuracy: 96.67%
Epoch 200:   Loss 0.3638,   Validation Accuracy: 96.67%
Epoch 300:   Loss 0.0072,   Validation Accuracy: 96.67%

Testing:  LR=0.01 ,  Batch=32
Epoch 0:   Loss 2.5952,   Validation Accuracy: 30.00%
Epoch 100:   Loss 0.1365,   Validation Accuracy: 100.00%
Epoch 200:   Loss 0.0600,   Validation Accuracy: 96.67%
Epoch 300:   Loss 0.0435,   Validation Accuracy: 96.67%

Testing:  LR=0.001 ,  Batch=8
Epoch 0:   Loss 1.2392,   Validation Accuracy: 30.00%
Epoch 100:   Loss 0.5963,   Validation Accuracy: 86.67%
Epoch 200:   Loss 0.1522,   Validation Accuracy: 100.00%
Epoch 300:   Loss 1.6811,   Validation Accuracy: 96.67%

Testing:  LR=0.001 ,  Batch=16
Epoch 0:   Loss 2.2079,   Validation Accuracy: 3.33%
Epoch 100:   Loss 0.4813,   Validation Accuracy: 93.33%
Epoch 200:   Loss 0.3295,   Validation Accuracy: 93.33%
Epoch 300:   Loss 0.3474,   Validation Accuracy: 96.67%

Testing:  LR=0.001 ,  Batch=32
Epoch 0:   Loss 1.8867,   Validation Accuracy: 6.67%
Epoch 100:   Loss 0.4335,   Validation Accuracy: 86.67%
Epoch 200:   Loss 0.3340,   Validation Accuracy: 100.00%
Epoch 300:   Loss 0.2512,   Validation Accuracy: 100.00%

 === Final Best Configuration ===
Config: LR=0.1, Batch=8, With accuracy = 100.00%
Epoch 0:   Loss 0.4586,   Validation Accuracy: 73.33%
Epoch 100:   Loss 0.0783,   Validation Accuracy: 93.33%
Epoch 200:   Loss 0.0077,   Validation Accuracy: 93.33%
Epoch 300:   Loss 0.1505,   Validation Accuracy: 93.33%
Epoch 400:   Loss 0.5045,   Validation Accuracy: 100.00%
Epoch 500:   Loss 0.9727,   Validation Accuracy: 96.67%
Epoch 600:   Loss 0.4230,   Validation Accuracy: 100.00%
Epoch 700:   Loss 0.0063,   Validation Accuracy: 100.00%
Epoch 800:   Loss 0.0051,   Validation Accuracy: 93.33%
Epoch 900:   Loss 0.0106,   Validation Accuracy: 100.00%
Epoch 1000:   Loss 0.0516,   Validation Accuracy: 100.00%
PS C:\Users\lubanadaabol\python_projects\AldaabolMiniNNproject> 
