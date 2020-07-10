#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:43:38 2020

@author: mahathir
"""
#%%
import numpy as np
import pandas as pd
import pickle

#%%
imputer = pickle.load(open('model/imputer.sav', 'rb'))
le = pickle.load(open('model/label_encoder.sav', 'rb'))
sc = pickle.load(open('model/standard_scaler.sav', 'rb'))

#%%
from tensorflow.keras.models import model_from_json
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model/model.h5")

#%%

dataset = pd.read_csv('test.csv')


#%%
X = dataset.iloc[:, [1,3,4,5,6,8,9]].values
X[:, 1] = le.transform(X[:, 1])
X[:, 2:3] = imputer.transform(X[:, 2:3])
X[:,6]=[0 if x is np.nan else 1 for x in X[:,6]]
X = sc.transform(X)

#%%
y_pred = loaded_model.predict(X)
y_pred = (y_pred > 0.5).astype(int)
#%%
ids = dataset.iloc[:, 0].values
import csv
with open('submit.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["PassengerId", "Survived"])
    
    for i in range(ids.shape[0]):
        writer.writerow([ids[i], y_pred[i,0]])