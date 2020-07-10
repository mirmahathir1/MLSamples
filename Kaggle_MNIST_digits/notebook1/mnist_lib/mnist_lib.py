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
