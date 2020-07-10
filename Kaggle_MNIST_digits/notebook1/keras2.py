%reset -f
import pandas as pd
import numpy as np
class mnist_kaggle:
  def __init__(self):
    dataset = pd.read_csv("dataset/train.csv")
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    y = dataset["label"].copy()
    X = dataset.drop(['label'], axis = 1)
    y = pd.get_dummies(y, columns=["label"], prefix="label" )

    #X=np.array(X)/255
    X=np.array(X)
    y=np.array(y)

    from sklearn.model_selection import train_test_split
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    self.curr_iteration=0
    
  def next_batch(self,batch_size):
    max_segments=int(self.X_train.shape[0]/batch_size)
    curr_segment=self.curr_iteration%max_segments
    batch_X=self.X_train[curr_segment*batch_size:(curr_segment+1)*batch_size:]
    batch_y=self.y_train[curr_segment*batch_size:(curr_segment+1)*batch_size:]
    self.curr_iteration+=1
    return (batch_X,batch_y)

mnist=mnist_kaggle()

X_train,X_test,y_train,y_test = (mnist.X_train,mnist.X_test,mnist.y_train,mnist.y_test)
#%% keras2.1-2.4
X_train=X_train.reshape((X_train.shape[0],28,28,1))
X_test=X_test.reshape((X_test.shape[0],28,28,1))

#%% keras2.5
X_train=X_train.reshape((X_train.shape[0],28,28,1))/255
X_test=X_test.reshape((X_test.shape[0],28,28,1))/255

#%% 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Dropout
#create model
model = Sequential()
#add model layers
model.add(Conv2D(32, kernel_size=6, activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same"))
model.add(Conv2D(64, kernel_size=6, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same"))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#%% keras2.1
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
#%% keras2.2
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20)
#%% keras2.3
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40)
#%% keras2.4
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=35)
#%% keras2.5
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)
#%%
model.save('models/keras2.5.h5')
#%%
dataset_submit = pd.read_csv("dataset/test.csv")
X_submit = dataset_submit.copy()
X_submit=np.array(X_submit).reshape(X_submit.shape[0],28,28,1)


prediction=model.predict(X_submit)

prediction_submit=np.zeros((prediction.shape[0],1), dtype=np.int32)
for i in range(prediction.shape[0]):
    maxi=prediction[i].argmax()
    prediction_submit[i]=maxi

import csv
with open('submit.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ImageId", "Label"])
    
    for i in range(prediction_submit.shape[0]):
        writer.writerow([i+1, prediction_submit[i][0]])


