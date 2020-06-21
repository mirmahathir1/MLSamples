#%%
#import tensorflow as tf
%reset -f
from house_price_libs import clean_minimize_encode,load_train_dataset,print_stats,load_test_dataset
import pandas as pd
import numpy as np

X,y = load_train_dataset()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 600, random_state = 0)
regressor.fit(X_train, y_train);

print("Dev: ")
predictions = regressor.predict(X_test)
print_stats(predictions,y_test)
print("Train: ")
predictions = regressor.predict(X_train)
print_stats(predictions,y_train)



X_submit,ids=load_test_dataset()

predictions_submit = regressor.predict(X_submit)


#%%
import csv
with open('submit.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Id", "SalePrice"])
    
    for i in range(ids.shape[0]):
        writer.writerow([ids[i], predictions_submit[i]])


