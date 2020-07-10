#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import keras as K
import pickle

# In[2]:


dataset = pd.read_csv('train.csv')


# In[3]:


X = dataset.iloc[:, [2,4,5,6,7,9,10]].values
y = dataset.iloc[:, 1].values


# In[4]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
pickle.dump(le, open('model/label_encoder.sav', 'wb'))


# In[5]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])

pickle.dump(imputer, open('model/imputer.sav', 'wb'))

# In[6]:


X[:,6]=[0 if x is np.nan else 1 for x in X[:,6]]


# In[7]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

pickle.dump(sc, open('model/standard_scaler.sav', 'wb'))

# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[9]:


ann = K.models.Sequential()
ann.add(K.layers.Dense(units=7, activation='relu'))
ann.add(K.layers.Dense(units=8, activation='relu'))
ann.add(K.layers.Dense(units=9, activation='relu'))
ann.add(K.layers.Dense(units=10, activation='relu'))
ann.add(K.layers.Dense(units=9, activation='relu'))
ann.add(K.layers.Dense(units=8, activation='relu'))
ann.add(K.layers.Dense(units=7, activation='relu'))
ann.add(K.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[10]:


ann.fit(X_train, y_train, epochs = 1000)


# In[11]:


model_json = ann.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
ann.save_weights("model/model.h5")


# In[12]:


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
