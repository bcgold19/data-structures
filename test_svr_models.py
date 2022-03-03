# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 20:30:10 2021

@author: victo
"""
import numpy as np
import pandas as pd

from my_stat_tools import *
from data_importing import *
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

X_train_normal,mu_xtr,sigma_xtr=standardised(X_train)
y_train_normal,mu_ytr,sigma_ytr=standardised(y_train)
X_test_normal,mu_xte,sigma_xte=standardised(X_test)
y_test_normal,mu_yte,sigma_xte=standardised(y_test)
### Hybrid model
# window =1, y_pred gained from the linear regression: and calculated with Expectation maximasation algorithm
estimator_training_=EMalgorithm(X_train,y_train)

B_normal=estimator_training_[2]
alpha_normal=estimator_training_[3]

y_pred=X_train@B_normal.T+np.tile(alpha_normal,(X_train.shape[0],1))
y_pred_test=X_test@B_normal.T+np.tile(alpha_normal,(X_test.shape[0],1))


delta_train=np.array(y_train)-np.array(y_pred)
delta_train_=delta_train[1:]
delta_lagged_train=delta_train[:-1]
X_train_=X_train[1:]
X_train_['lagged_res_0']=delta_lagged_train[:,0]


## test data prep for asset 1
delta_test=np.array(y_test_normal)-np.array(y_pred_test)
delta_test_=delta_test[1:]
delta_lagged_test=delta_test[:-1]
X_test_=X_test[1:]
X_test_['lagged_res_0']=delta_lagged_test[:,0]


score=make_scorer(mean_absolute_error,greater_is_better=False)
kernel = ['rbf']



C = [0.1,1,2,2.5,10,100,200,300,400,500]
gamma = [10,0.1,300,400,500,600] # find threshold fit best, 
epsilon = [0.01,0.1]
shrinking = [True,False]
svm_grid = {'kernel':kernel,'C':C,'gamma':gamma,'epsilon':epsilon,'shrinking':shrinking}
SVM = SVR()
svm_search_hybrid = RandomizedSearchCV(SVM,svm_grid,cv = 5,n_jobs = -1,scoring=score)
svm_search_hybrid.fit(X_train_,delta_train_[:,0])

#svm_search_hybrid = SVR(kernel="linear", C=100, gamma="auto")
#svm_search_hybrid.fit(X_train_,delta_train_[:,3])


## fitting window=1 

delta_zero_hat=svm_search_hybrid.predict(X_train_)## error prediction on the training set
delta_zero_hat_test=svm_search_hybrid.predict(X_test_) ## error prediction on the test set
y_hib_tr=delta_zero_hat+y_pred.iloc[1:,0]
y_hib_te=delta_zero_hat_test+y_pred_test.iloc[1:,0]


print(r2_score(y_train.iloc[1:,0],y_hib_tr))
print(r2_score(y_test.iloc[1:,0],y_hib_te))

print("Mean Absolute value for train set: ",mean_absolute_error(y_train_normal.iloc[1:,3],y_hib_tr))
print("Mean Absolute value for test set: ",mean_absolute_error(y_test_normal.iloc[1:,3],y_hib_te))