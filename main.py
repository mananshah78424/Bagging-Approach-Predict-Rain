#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:43:15 2024

@author: mananshah
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
     

train_X = pd.read_parquet("test_inputs.parquet")
train_Y = pd.read_parquet('test_targets.parquet')["RainTomorrow"]

print('train_inputs:', train_X.shape)
print('train_targets:', train_Y.shape)

numeric_cols = train_X.select_dtypes(include=np.number).columns.tolist()

X = train_X[numeric_cols]
train_inputs, test_inputs ,train_target ,test_target = train_test_split(X, train_Y, test_size=0.2, random_state=42)

print('train_inputs:', train_inputs.shape)
print('train_targets:', train_target.shape)

print('test_inputs:', test_inputs.shape)
print('test_targets:', test_target.shape)

# svc = SVC(C = 0.01, kernel= "sigmoid", random_state=42)
# svc.fit(train_inputs,train_target)
# y_pred = svc.predict(test_inputs)

# print("SVC accuracy",accuracy_score(test_target,y_pred))

# #Not such a great score, lets try bagging approach - SVC accuracy 0.7533718689788054


# bagging = BaggingClassifier(
#     base_estimator=SVC(),
#     n_estimators=500,
#     max_samples=0.5,
#     bootstrap=True,
#     random_state=42
# )
    
# bagging.fit(train_inputs,train_target)
# y_pred = bagging.predict(test_inputs)
# print("Bagging using SVC",accuracy_score(test_target,y_pred))
# #Fairly decent score - Bagging using SVC 0.8073217726396917



