
#import tensorflow as tf
#%%
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
#%%

def clean(dataset_orig):
    dataset_clean = dataset_orig.copy()
    
    #completely drop these columns
    dataset_clean = dataset_clean.drop(['Utilities','Id',"Street","MiscVal"], axis=1)
    
    #Replace all nan with None
    for col in ["Alley",
                "MasVnrType",
                "BsmtQual",
                "BsmtCond",
                "BsmtExposure",
                "BsmtFinType1",
                "BsmtFinType2",
                "FireplaceQu",
                "GarageType",
                "GarageFinish",
                "GarageQual",
                "GarageCond",
                "PoolQC",
                "Fence",
                "MiscFeature",
                ]:    
        dataset_clean[col] = dataset_clean[col].fillna("None")
    
    #replace missing values or outliers with most occuring
    
    dataset_clean['Functional']=dataset_clean['Functional'].replace(np.nan, 'Typ')
    
    dataset_clean['KitchenQual']=dataset_clean['KitchenQual'].replace(np.nan, 'TA')
    
    dataset_clean['MSZoning']=dataset_clean['MSZoning'].replace(np.nan, 'RL')
    
    dataset_clean['SaleType']=dataset_clean['SaleType'].replace(np.nan, 'WD')
    
    dataset_clean['LotConfig']=dataset_clean['LotConfig'].replace(np.nan, 'Inside')
    
    dataset_clean['Electrical']=dataset_clean['Electrical'].replace(np.nan, 'SBrkr')
    
    dataset_clean['Exterior2nd']=dataset_clean['Exterior2nd'].replace({np.nan: 'VinylSd',"Other":'VinylSd',"ImStucc":'VinylSd',"Stone":'VinylSd'})
    dataset_clean['Exterior1st']=dataset_clean['Exterior1st'].replace({np.nan: 'VinylSd',"Other":'VinylSd',"ImStucc":'VinylSd',"Stone":'VinylSd'})
    
    #fix typos
    dataset_clean['Exterior1st']=dataset_clean['Exterior1st'].replace({"Wd Shng":"WdShing","CmentBd":"CemntBd","Brk Cmn":"BrkComm"})
    dataset_clean['Exterior2nd']=dataset_clean['Exterior2nd'].replace({"Wd Shng":"WdShing","CmentBd":"CemntBd","Brk Cmn":"BrkComm"})
    
    
    #replace missing values with median
    imputer_GarageYrBlt = SimpleImputer(missing_values = np.nan, strategy = 'median')
    dataset_clean["GarageYrBlt"] = imputer_GarageYrBlt.fit_transform(dataset_clean[["GarageYrBlt"]]).ravel()
    
    #replace missing value with zero
    dataset_clean['MasVnrArea']=dataset_clean['MasVnrArea'].replace(np.nan,0.0)
    dataset_clean['LotFrontage']=dataset_clean['LotFrontage'].replace(np.nan,0.0)
    
    dataset_clean['BsmtFinSF1']=dataset_clean['BsmtFinSF1'].replace(np.nan,0.0)
    dataset_clean['BsmtFinSF2']=dataset_clean['BsmtFinSF2'].replace(np.nan,0.0)
    dataset_clean['BsmtUnfSF']=dataset_clean['BsmtUnfSF'].replace(np.nan,0.0)
    dataset_clean['TotalBsmtSF']=dataset_clean['TotalBsmtSF'].replace(np.nan,0.0)
    dataset_clean['BsmtFullBath']=dataset_clean['BsmtFullBath'].replace(np.nan,0.0)
    dataset_clean['BsmtHalfBath']=dataset_clean['BsmtHalfBath'].replace(np.nan,0.0)
    dataset_clean['GarageCars']=dataset_clean['GarageCars'].replace(np.nan,0.0)
    dataset_clean['GarageArea']=dataset_clean['GarageArea'].replace(np.nan,0.0)
    
    return dataset_clean

def minimize_encode(dataset_clean):
    dataset = dataset_clean.copy()
    
    labelDictionary={
        #Encoding ordinal data in this section
        "CentralAir":{"Y":1,"N":0},
        "LotShape":{"Reg":4,"IR1":3,"IR2":2,"IR3":1},
        "LandContour":{"Lvl":4,"Bnk":3,"HLS":2,"Low":1},
        "LandSlope":{"Gtl":3,"Mod":2,"Sev":1},
        "ExterQual":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1},
        "BsmtQual":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0},
        "ExterCond":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1},
        "BsmtCond":{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0},
        "BsmtExposure":{"Gd":4,"Av":3,"Mn":2,"No":1,"None":0},
        'HeatingQC':{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1},
        'KitchenQual':{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1},
        'Functional':{"Typ":8,"Min1":7,"Min2":6,"Mod":5,"Maj1":4,"Maj2":3,"Sev":2,"Sal":1},
        'FireplaceQu':{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0},
        'GarageFinish':{"Fin":3,"RFn":2,"Unf":1,"None":0},
        'GarageQual':{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0},
        'GarageCond':{"Ex":5,"Gd":4,"TA":3,"Fa":2,"Po":1,"None":0},
        'PavedDrive':{"Y":3,"P":2,"N":1},
        'PoolQC':{"Ex":4,"Gd":3,"TA":2,"Fa":1,"None":0},
        'Fence':{"GdPrv":4,"MnPrv":3,"GdWo":2,"MnWw":1,"None":0},
        'BsmtFinType1':{"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"None":0},
        'BsmtFinType2':{"GLQ":6,"ALQ":5,"BLQ":4,"Rec":3,"LwQ":2,"Unf":1,"None":0},  
    
        #Minimize amount of categories in this section
        'HouseStyle':{"2.5Fin":"2.5Unf"},
        'Foundation':{"Stone":"Other","Wood":"Other","Slab":"Other"},
        'RoofMatl':{"ClyTile":"Other","Membran":"Other","Metal":"Other","Roll":"Other","WdShake":"Other","WdShngl":"Other"},
        'RoofStyle':{"Flat":"Other","Gambrel":"Other","Mansard":"Other","Shed":"Other"},
        'LotConfig':{"FR3":"FR","FR2":"FR"},
        'SaleType':{"CWD":"WD","VWD":"WD","ConLw":"Con","ConLI":"Con","ConLD":"Con","Oth":"WD"},
        'MoSold':{1:"1stQuarter",2:"1stQuarter",3:"1stQuarter",4:"2ndQuarter",5:"2ndQuarter",6:"2ndQuarter",7:"3rdQuarter",8:"3rdQuarter",9:"3rdQuarter",10:"4thQuarter",11:"4thQuarter",12:"4thQuarter"},
        'MiscFeature':{"Gar2":"Othr","Shed":"Othr","Shed":"Othr","TenC":"Othr"},
        'Electrical':{"FuseA":"Fuse","FuseF":"Fuse","FuseP":"Fuse","Mix":"Fuse"},
        'Heating':{"Floor":"Other","GasW":"Other","Grav":"Other","OthW":"Other","Wall":"Other"},
        'Condition1':{"RRNe":"RRe","RRAe":"RRe","RRNn":"RRn","RRAn":"RRn","PosA":"Pos","PosN":"Pos"},
        'Condition2':{"RRNe":"RRe","RRAe":"RRe","RRNn":"RRn","RRAn":"RRn","PosA":"Pos","PosN":"Pos"}
        }
    
    for key,value in labelDictionary.items():
        dataset[key]=dataset[key].replace(value)
    return dataset

def one_hot_encode(dataset):
    columns_for_onehotencode=[
        "SaleCondition",
        "YrSold",
        "MSZoning",
        "Alley",
        "Neighborhood",
        "GarageType",
        "BldgType",
        "HouseStyle",
        "MasVnrType",
        
        "Foundation",
        "RoofMatl",
        "RoofStyle",
        "LotConfig",
        "SaleType",
        "MoSold",
        "MiscFeature",
        "Electrical",
        "Heating",
        "Exterior1st",
        "Exterior2nd",
        "Condition1",
        "Condition2"
        ]
    
    
    # from sklearn.preprocessing import LabelEncoder
    # lb_make = LabelEncoder()
    # train_clean["Condition1"] = lb_make.fit_transform(train_clean["Condition1"])
    
    for col in columns_for_onehotencode:
        unique_values = dataset[col].unique()
        unique_values.sort()
        dataset = pd.get_dummies(dataset, columns=[col], prefix=[col] )
        dataset = dataset.drop([col+"_"+str(unique_values[0])], axis=1)
        
    return dataset

def clean_minimize_encode(csv_path):
    dataset_orig= pd.read_csv(csv_path)
    dataset_clean = clean(dataset_orig)
    dataset = minimize_encode(dataset_clean)
    #%%
    
    
    #%%
    unique_vals_Exterior = np.union1d(dataset["Exterior1st"].unique(), dataset["Exterior2nd"].unique())
    unique_vals_Condition = np.union1d(dataset["Condition1"].unique(), dataset["Condition2"].unique())
    
    #%%
    
    dataset = one_hot_encode(dataset)
    
    
    #Compress two main columns into one
    for col in unique_vals_Exterior:
        if "Exterior1st_"+col in dataset.columns:
            dataset["Exterior_"+col]  = dataset["Exterior1st_"+col] + dataset["Exterior2nd_"+col]
            dataset = dataset.drop(["Exterior1st_"+col,"Exterior2nd_"+col], axis=1)
    
    if not "Condition2_RRn" in dataset.columns:
        dataset["Condition2_RRn"]=0

    if not "Condition2_RRe" in dataset.columns:
        dataset["Condition2_RRe"]=0
    
    for col in unique_vals_Condition:
        if "Condition1_"+col in dataset.columns:
            dataset["Condition_"+col]  = dataset["Condition1_"+col] + dataset["Condition2_"+col]
    
    #dataset = drop_columns(dataset)
    
    return (dataset_orig,dataset_clean,dataset)

def load_train_dataset():
    _,_,train = clean_minimize_encode("dataset/train.csv")
    y = np.array(train['SalePrice'])

    X= train.drop('SalePrice', axis = 1)
    
    #feature_list = list(X.columns)
    
    X = np.array(X)
    return (X,y)
def load_test_dataset():
    test_orig,_,test = clean_minimize_encode("dataset/test.csv")
    return (np.array(test),test_orig["Id"])
    
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
    
    print(np.sqrt(np.sum(np.square(np.log(predictions)-np.log(test_labels)))/predictions.shape[0]))
        
    print("________________________")


# Variable: OverallQual          Importance: 0.57
# Variable: GrLivArea            Importance: 0.12
# Variable: TotalBsmtSF          Importance: 0.04
# Variable: BsmtFinSF1           Importance: 0.03
# Variable: GarageCars           Importance: 0.03
# Variable: 1stFlrSF             Importance: 0.02
# Variable: GarageArea           Importance: 0.02
# Variable: LotFrontage          Importance: 0.01
# Variable: LotArea              Importance: 0.01
# Variable: YearBuilt            Importance: 0.01
# Variable: YearRemodAdd         Importance: 0.01
# Variable: MasVnrArea           Importance: 0.01
# Variable: BsmtQual             Importance: 0.01
# Variable: BsmtUnfSF            Importance: 0.01
# Variable: 2ndFlrSF             Importance: 0.01
# Variable: KitchenQual          Importance: 0.01
# Variable: TotRmsAbvGrd         Importance: 0.01
# Variable: WoodDeckSF           Importance: 0.01
# Variable: OpenPorchSF          Importance: 0.01

    