#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 22:00:48 2020

@author: mahathir
"""
%reset -f

import numpy as np
import pandas as pd
from mnist_lib.mnist_lib import mnist_kaggle

mnist=mnist_kaggle()

X_train,X_test,y_train,y_test = (mnist.X_train,mnist.X_test,mnist.y_train,mnist.y_test)
X_train=X_train.reshape((X_train.shape[0],28,28,1))/255
X_test=X_test.reshape((X_test.shape[0],28,28,1))/255

print(X_train.shape)