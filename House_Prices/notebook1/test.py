#%%
#import tensorflow as tf

#%%
%reset -f
import numpy as np
import pandas as pd
from house_price_libs import clean_minimize_encode

features_orig,features_clean,features = clean_minimize_encode("dataset/train.csv")
# features_orig,features_clean,features = load_train_set()

# columns_for_onehotencode=[
#         "SaleCondition",
#         "YrSold",
#         "MSZoning",
#         "Alley",
#         "Neighborhood",
#         "GarageType",
#         "BldgType",
#         "HouseStyle",
#         "MasVnrType",
        
#         "Foundation",
#         "RoofMatl",
#         "RoofStyle",
#         "LotConfig",
#         "SaleType",
#         "MoSold",
#         "MiscFeature",
#         "Electrical",
#         "Heating",
#         "Exterior1st",
#         "Exterior2nd",
#         "Condition1",
#         "Condition2"
#         ]

# from sklearn.preprocessing import LabelEncoder
# for column in columns_for_onehotencode:
#     lb_make = LabelEncoder()
#     features[column] = lb_make.fit_transform(features[column])
    
#%%

# from sklearn.preprocessing import LabelEncoder
# lb_make = LabelEncoder()
# features["Neighborhood"] = lb_make.fit_transform(features["Neighborhood"])

# columns_for_onehotencode=[
#         "SaleCondition",
#         "YrSold",
#         "MSZoning",
#         "Alley",
#         "Neighborhood",
#         "GarageType",
#         "BldgType",
#         "HouseStyle",
#         "MasVnrType",
#         "Foundation",
#         "RoofMatl",
#         "RoofStyle",
#         "LotConfig",
#         "SaleType",
#         "MoSold",
#         "MiscFeature",
#         "Electrical",
#         "Heating",
#         "Exterior1st",
#         "Exterior2nd",
#         "Condition1",
#         "Condition2"
#         ]

# columns_for_labelencode=[
#         "SaleCondition",
#         "YrSold",
#         "MSZoning",
#         "Alley",
#         "Neighborhood",
#         "GarageType",
#         "BldgType",
#     ]

# columns_for_onehotencode=[item for item in columns_for_onehotencode if item not in columns_for_labelencode]

# label_dictionary=dict()

# for column in columns_for_labelencode:
#     lb_make = LabelEncoder()
#     features[column] = lb_make.fit_transform(features[column])
#     label_dictionary[column]=lb_make


#%%
labels = np.array(features['SalePrice'])
#labels = labels.reshape(labels.shape[0],1)

print(labels.shape)

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('SalePrice', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)



# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 0)


# The baseline prediction
baseline_preds = labels.mean()
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))



def print_stats(predictions,test_labels):
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'Dollar.')
        
    
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2),'%.')
        
    print("________________________")

#%%

# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# train_features = sc_X.fit_transform(train_features)
# test_features = sc_X.transform(test_features)


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 600, random_state = 0)
rf.fit(train_features, train_labels);

predictions = rf.predict(test_features)
print_stats(predictions,test_labels)



#%%
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(train_features)
X_test = sc_X.transform(test_features)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, train_labels)

predictions = regressor.predict(X_test)
print_stats(predictions,test_labels)
#%%
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(train_features)
X_test = sc_X.transform(test_features)

# X_train = train_features
# X_test = test_features

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, train_labels)

predictions = regressor.predict(X_test)
print_stats(predictions,test_labels)
#%%
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(train_features)
X_test = sc_X.transform(test_features)

# X_train = train_features
# X_test = test_features

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, train_labels)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, train_labels)


predictions = lin_reg_2.predict(poly_reg.fit_transform(X_test))
print_stats(predictions,test_labels)
#%%
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(train_features, train_labels)

predictions =regressor.predict(test_features)
print_stats(predictions,test_labels)
#%%
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(train_features)
X_test = sc_X.transform(test_features)
# X_train = train_features
# X_test = test_features

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, train_labels)

predictions =classifier.predict(test_features)
print_stats(predictions,test_labels)
#%%
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(train_features)
X_test = sc_X.transform(test_features)

# X_train = train_features
# X_test = test_features

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, train_labels)

predictions =classifier.predict(test_features)
print_stats(predictions,test_labels)
#%%
#%%
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


#%%




# #test_clean = clean(pd.read_csv('dataset/train.csv'))
# dataset_clean = clean(pd.read_csv('dataset/test.csv'))

# #print(dataset_clean.groupby('SaleType').count())

# dataset = minimize_encode(dataset_clean)
# #test = minimize_encode(test_clean)


# unique_vals_Exterior = np.union1d(dataset["Exterior1st"].unique(), dataset["Exterior2nd"].unique())
# unique_vals_Condition = np.union1d(dataset["Condition1"].unique(), dataset["Condition2"].unique())

# dataset = one_hot_encode(dataset)

# #%%
# #Compress two main columns into one
# for col in unique_vals_Exterior:
#     if "Exterior1st_"+col in dataset.columns:
#         dataset["Exterior_"+col]  = dataset["Exterior1st_"+col] + dataset["Exterior2nd_"+col]
#         dataset = dataset.drop(["Exterior1st_"+col,"Exterior2nd_"+col], axis=1)

# if not "Condition2_RRn" in dataset.columns:
#     dataset["Condition2_RRn"]=0

# if not "Condition2_RRe" in dataset.columns:
#     dataset["Condition2_RRe"]=0

# for col in unique_vals_Condition:
#     if "Condition1_"+col in dataset.columns:
#         dataset["Condition_"+col]  = dataset["Condition1_"+col] + dataset["Condition2_"+col]
#         dataset = dataset.drop(["Condition1_"+col,"Condition2_"+col], axis=1)
#%%




