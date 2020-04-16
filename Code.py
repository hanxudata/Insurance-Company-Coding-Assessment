import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib


# load train data
tran_data=pd.read_csv('exercise_03_train.csv')

# load test data
test_data=pd.read_csv('exercise_03_test.csv')

# concatenate train data with test data
data=pd.concat([tran_data,test_data]).reset_index(drop=True)

# Data Preprocessing
# x35 map
x35_map={'monday':2,'tuesday':3,'wed':4,'wednesday':4,'thur':5,'thurday':5,'fri':6,'friday':6}

# x68 map
x68_map={'January':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'July':7,'Aug':8,'sept.':9,'Oct':10,'Nov':11,'Dev':12}

# replace strings for x35 and x68 
data['x35']=data['x35'].map(x35_map)
data['x68']=data['x68'].map(x68_map)

# remove special characters and turn strings to float for x41 and x45
data['x41']=data['x41'].str.replace('$','').astype('float64')
data['x45']=data['x45'].str.replace('%','').astype('float64')

# one-hot encoding for x34 and x93 strings
data_x34=pd.get_dummies(data['x34'],prefix='x34')
data_x93=pd.get_dummies(data['x93'],prefix='x93')

# concatenate one-hot encoding data with original data
data.drop(['x34','x93'],axis=1,inplace=True)
data=pd.concat([data,data_x34,data_x93],axis=1)

#Replace missing data with mean value
for column in list(data.columns[data.isnull().sum() > 0]):
    mean_val = data[column].mean()
    data[column].fillna(mean_val, inplace=True)


# Model Training    
train_length=len(tran_data)
columns = [x for x in data.columns if x not in ['y']]

mm = MinMaxScaler()
data.loc[:,columns] = mm.fit_transform(data[columns])


x=data.loc[0:train_length-1,columns]
y=data.loc[0:train_length-1,'y']

# split data
x_test=data.loc[train_length:,columns]
x_train,x_vali,y_train,y_vali=train_test_split(x,y,test_size=0.3,random_state=2020)



# train RandomForestClassifier
rf=RandomForestClassifier(n_estimators=200,random_state=2020,n_jobs=-1)
rf.fit(x_train,y_train)
y_vali_pred=rf.predict_proba(x_vali)[:,1]
print("RandomForest auc:{:.2f}".format(roc_auc_score(y_vali,y_vali_pred)))
x_result=x_test.copy()
x_result['pred']=rf.predict_proba(x_test)[:,1]
x_result['pred'].to_csv('results1.csv',index=False,header=False)



# train LGBMClassifier
gbm = lgb.LGBMClassifier()
gbm.fit(x_train,y_train,eval_set=[(x_vali, y_vali)],eval_metric='auc',early_stopping_rounds=5,verbose=0)
y_vali_pred = gbm.predict_proba(x_vali, num_iteration=gbm.best_iteration_)[:,1]
print("LGBM auc:{:.2f}".format(roc_auc_score(y_vali,y_vali_pred)))
x_result=x_test.copy()
x_result['pred']=gbm.predict_proba(x_test)[:,1]
x_result['pred'].to_csv('results2.csv',index=False,header=False)



# # RandomForest parameter search
# param_grid = {'n_estimators':[10,50,100]}
# rf = GridSearchCV(estimator = rf, param_grid = param_grid, scoring='roc_auc',cv=5)
# rf.fit(x,y)
# print('Best parameters found by grid search are:',rf.best_params_)


# # gbm parameter search
# param_grid = {'n_estimators': [20,50,100]}
# gbm = GridSearchCV(gbm, param_grid)
# gbm.fit(x_train,y_train,eval_set=[(x_vali, y_vali)],eval_metric='auc',early_stopping_rounds=5,verbose=0)
# print('Best parameters found by grid search are:', gbm.best_params_)

