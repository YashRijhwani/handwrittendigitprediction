# handwrittendigitprediction
Hand Written Digit Prediction
Objective
A handwritten digit prediction machine learning model aims to accurately recognize and classify handwritten digits (0-9) from images. This enhances the efficiency and accuracy of digit recognition tasks in various applications, such as automated data entry and document processing. By learning patterns in handwritten data, the model can generalize to recognize new, unseen digits with high precision. Overall, the objective is to streamline tasks that involve digit recognition, improving productivity and reducing errors.

Import Library

import pandas as pd
     

import numpy as np
     

import matplotlib.pyplot as plt
     
Import Data

from sklearn.datasets import load_digits
     

hwdp = load_digits()
     

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, hwdp.images, hwdp.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)
     

Data Preprocessing

hwdp.images.shape
     
(1797, 8, 8)

hwdp.images[0]
     
array([[ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.],
       [ 0.,  0., 13., 15., 10., 15.,  5.,  0.],
       [ 0.,  3., 15.,  2.,  0., 11.,  8.,  0.],
       [ 0.,  4., 12.,  0.,  0.,  8.,  8.,  0.],
       [ 0.,  5.,  8.,  0.,  0.,  9.,  8.,  0.],
       [ 0.,  4., 11.,  0.,  1., 12.,  7.,  0.],
       [ 0.,  2., 14.,  5., 10., 12.,  0.,  0.],
       [ 0.,  0.,  6., 13., 10.,  0.,  0.,  0.]])

hwdp.images[0].shape
     
(8, 8)

len(hwdp.images)
     
1797

n_samples = len(hwdp.images)
data = hwdp.images.reshape((n_samples, -1))
     

data[0]
     
array([ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.,  0.,  0., 13., 15., 10.,
       15.,  5.,  0.,  0.,  3., 15.,  2.,  0., 11.,  8.,  0.,  0.,  4.,
       12.,  0.,  0.,  8.,  8.,  0.,  0.,  5.,  8.,  0.,  0.,  9.,  8.,
        0.,  0.,  4., 11.,  0.,  1., 12.,  7.,  0.,  0.,  2., 14.,  5.,
       10., 12.,  0.,  0.,  0.,  0.,  6., 13., 10.,  0.,  0.,  0.])

data[0].shape
     
(64,)

data.shape
     
(1797, 64)
Scaling Image Data

data.min()
     
0.0

data.max()
     
16.0

data = data/16
     

data.min()
     
0.0

data.max()
     
1.0

data[0]
     
array([0.    , 0.    , 0.3125, 0.8125, 0.5625, 0.0625, 0.    , 0.    ,
       0.    , 0.    , 0.8125, 0.9375, 0.625 , 0.9375, 0.3125, 0.    ,
       0.    , 0.1875, 0.9375, 0.125 , 0.    , 0.6875, 0.5   , 0.    ,
       0.    , 0.25  , 0.75  , 0.    , 0.    , 0.5   , 0.5   , 0.    ,
       0.    , 0.3125, 0.5   , 0.    , 0.    , 0.5625, 0.5   , 0.    ,
       0.    , 0.25  , 0.6875, 0.    , 0.0625, 0.75  , 0.4375, 0.    ,
       0.    , 0.125 , 0.875 , 0.3125, 0.625 , 0.75  , 0.    , 0.    ,
       0.    , 0.    , 0.375 , 0.8125, 0.625 , 0.    , 0.    , 0.    ])
Train Text Split Data

from sklearn.model_selection import train_test_split
     

X_train, X_test, y_train, y_test = train_test_split(data, hwdp.target, test_size=0.3)
     

X_train.shape, X_test.shape, y_train.shape, y_test.shape
     
((1257, 64), (540, 64), (1257,), (540,))
Random Forest Model

from sklearn.ensemble import RandomForestClassifier
     

rf = RandomForestClassifier()
     

rf.fit(X_train, y_train)
     
RandomForestClassifier()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
Predict Test Data

y_pred = rf.predict(X_test)
     

y_pred
     
array([5, 2, 8, 0, 8, 8, 4, 7, 0, 0, 9, 1, 9, 9, 5, 7, 0, 5, 5, 9, 7, 9,
       2, 3, 0, 6, 4, 9, 1, 0, 3, 7, 6, 0, 0, 2, 7, 7, 6, 4, 1, 9, 3, 3,
       5, 2, 0, 2, 8, 8, 8, 4, 0, 3, 8, 7, 7, 8, 3, 4, 3, 0, 6, 2, 7, 7,
       8, 2, 7, 6, 1, 3, 8, 1, 9, 8, 1, 2, 6, 6, 6, 1, 9, 0, 8, 9, 0, 6,
       7, 3, 8, 4, 4, 6, 6, 7, 8, 6, 5, 2, 9, 9, 2, 3, 7, 3, 0, 8, 2, 2,
       7, 0, 2, 0, 4, 4, 2, 2, 1, 8, 7, 1, 0, 7, 7, 5, 2, 3, 1, 8, 0, 7,
       6, 4, 4, 4, 0, 2, 2, 2, 3, 6, 7, 0, 1, 4, 7, 0, 6, 1, 2, 6, 1, 6,
       0, 4, 9, 3, 1, 7, 7, 5, 8, 9, 4, 3, 3, 2, 2, 6, 2, 6, 3, 3, 3, 5,
       6, 5, 5, 6, 3, 0, 8, 0, 6, 9, 4, 2, 2, 4, 1, 6, 1, 1, 9, 4, 4, 0,
       1, 4, 4, 3, 0, 0, 7, 6, 2, 8, 2, 9, 2, 0, 7, 9, 0, 2, 5, 5, 6, 9,
       4, 2, 8, 3, 1, 8, 3, 8, 6, 6, 7, 1, 4, 4, 4, 5, 7, 7, 2, 5, 6, 2,
       8, 4, 9, 5, 0, 4, 5, 3, 1, 3, 7, 5, 3, 1, 1, 6, 6, 9, 3, 1, 3, 6,
       7, 1, 5, 4, 4, 3, 5, 2, 1, 9, 6, 5, 2, 9, 0, 5, 2, 0, 4, 5, 4, 3,
       3, 7, 3, 4, 1, 3, 9, 0, 9, 2, 8, 2, 8, 7, 1, 6, 3, 5, 3, 3, 0, 1,
       8, 3, 8, 1, 9, 6, 7, 9, 6, 6, 7, 8, 2, 0, 2, 9, 8, 5, 2, 7, 4, 3,
       6, 4, 1, 6, 3, 7, 3, 1, 6, 7, 3, 4, 5, 9, 6, 0, 0, 1, 4, 1, 7, 2,
       7, 8, 6, 9, 3, 7, 3, 8, 8, 7, 2, 9, 9, 0, 4, 7, 2, 7, 0, 3, 3, 3,
       1, 4, 6, 5, 3, 5, 6, 8, 9, 4, 1, 1, 3, 0, 8, 9, 1, 2, 6, 7, 0, 2,
       4, 7, 5, 3, 4, 1, 8, 5, 4, 4, 6, 6, 8, 1, 5, 0, 4, 5, 0, 8, 1, 1,
       7, 0, 3, 4, 0, 8, 5, 4, 4, 6, 4, 9, 2, 2, 6, 5, 5, 7, 4, 1, 8, 4,
       1, 4, 9, 6, 3, 4, 9, 1, 4, 8, 0, 1, 2, 8, 2, 5, 1, 5, 0, 5, 4, 0,
       9, 6, 9, 5, 4, 8, 3, 0, 8, 0, 5, 1, 4, 4, 9, 8, 0, 0, 8, 1, 8, 5,
       9, 7, 3, 7, 3, 1, 9, 0, 2, 6, 6, 7, 1, 5, 6, 2, 7, 8, 1, 0, 9, 2,
       4, 9, 8, 6, 4, 8, 0, 1, 1, 2, 3, 8, 5, 8, 7, 4, 2, 2, 6, 7, 7, 3,
       0, 6, 3, 1, 0, 8, 4, 3, 9, 7, 9, 7])
Model Accuracy

from sklearn.metrics import confusion_matrix, classification_report
     

confusion_matrix(y_test, y_pred)
     
array([[57,  0,  0,  0,  1,  0,  0,  0,  0,  0],
       [ 0, 55,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0, 54,  0,  0,  0,  0,  0,  0,  0],
       [ 0,  0,  0, 56,  0,  0,  0,  2,  2,  0],
       [ 0,  0,  0,  0, 60,  0,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0, 42,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0,  0, 56,  0,  1,  0],
       [ 0,  0,  0,  0,  0,  0,  0, 54,  1,  0],
       [ 0,  0,  1,  2,  0,  0,  0,  0, 49,  1],
       [ 0,  0,  0,  0,  0,  1,  0,  0,  0, 45]])

print(classification_report(y_test, y_pred))
     
              precision    recall  f1-score   support

           0       1.00      0.98      0.99        58
           1       1.00      1.00      1.00        55
           2       0.98      1.00      0.99        54
           3       0.97      0.93      0.95        60
           4       0.98      1.00      0.99        60
           5       0.98      1.00      0.99        42
           6       1.00      0.98      0.99        57
           7       0.96      0.98      0.97        55
           8       0.92      0.92      0.92        53
           9       0.98      0.98      0.98        46

    accuracy                           0.98       540
   macro avg       0.98      0.98      0.98       540
weighted avg       0.98      0.98      0.98       540

Explanation
A handwritten digit prediction machine learning model is designed to recognize and classify digits (0-9) from images of handwritten numbers. It learns from a large dataset of labeled digit images, understanding the unique features of each digit. When given a new, unseen handwritten digit, the model uses its learned knowledge to accurately predict the digit. This process involves analyzing the pixel patterns and structures in the image. Such models are useful in applications like automated data entry, digitizing handwritten notes, and verifying handwritten digits in various forms.

We are using the random forest classifier for this model.
