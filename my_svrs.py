# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 20:10:05 2021

@author: Beomcheol Kim
"""
import numpy as np
import pandas as pd
from my_stat_tools import *
from sklearn.metrics import make_scorer
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def hybrid_SVR_entire(X_tr,y_tr,window=1):
    # Get the linear estimator
    linear_reg=EMalgorithm(X_tr,y_tr)

    B_normal=linear_reg[2]
    alpha_normal=linear_reg[3]

    y_pred=X_tr@B_normal.T+np.tile(alpha_normal,(X_tr.shape[0],1))

    # for train set
    
    delta_train=np.array(y_tr)-np.array(y_pred)
    delta_train_=delta_train[window:]
    delta_lagged_train=delta_train[:-window]
    X_train_=X_tr[window:]

    
    # looping
    SVRs=[]
    predicted_train=[]
    residual=[]

    for i in range(y_tr.shape[1]):
        #i=int(i)
        
        
        tag_name=str('lagged_resuldual')+str(i)
        my_cols=[]
        for k in X_tr.columns:
            my_cols.append(k)
        my_cols.append(tag_name)
    
        X_train_[tag_name]=delta_lagged_train[:,i]
        

        score=make_scorer(mean_absolute_error,greater_is_better=False)
        kernel = ['rbf']
        C = [0.1,1,2,2.5,10,100,200,300,400,500]
        gamma = [10,0.1,300,400,500,600] # find threshold fit best, 
        epsilon = [0.01,0.1]
        shrinking = [True,False]
        svm_grid = {'kernel':kernel,'C':C,'gamma':gamma,'epsilon':epsilon,'shrinking':shrinking}
        SVM = SVR()
        svm_search_hybrid = RandomizedSearchCV(SVM,svm_grid,cv = 5,n_jobs = -1,scoring=score)
        svm_search_hybrid.fit(X_train_.loc[:,my_cols],delta_train_[:,i])
        
        SVRs.append(svm_search_hybrid)
        
        delta_zero_hat=svm_search_hybrid.predict(X_train_.loc[:,my_cols])
         
        y_hib_tr=delta_zero_hat+y_pred.iloc[window:,i]
        predicted_train.append(y_hib_tr)
        residual.append(delta_zero_hat)
        if(i%80==0):
            print(str(i)+" th asset fitting done")
    return SVRs,np.array(predicted_train),np.array(residual),linear_reg

def hybrid_SVR_(X_tr,y_tr,X_te,y_te,window=1):
    # Get the linear estimator
    estimator_training_=EMalgorithm(X_tr,y_tr)

    B_normal=estimator_training_[2]
    alpha_normal=estimator_training_[3]

    y_pred=X_tr@B_normal.T+np.tile(alpha_normal,(X_tr.shape[0],1))
    y_pred_test=X_te@B_normal.T+np.tile(alpha_normal,(X_te.shape[0],1))
    
    
    
    # for train set
    
    delta_train=np.array(y_tr)-np.array(y_pred)
    delta_train_=delta_train[window:]
    delta_lagged_train=delta_train[:-window]
    X_train_=X_tr[window:]
    
    # for test set
    delta_test=np.array(y_te)-np.array(y_pred_test)
    delta_test_=delta_test[window:]
    delta_lagged_test=delta_test[:-window]
    X_test_=X_te[window:]
    
    # looping
    SVRs=[]
    predicted_train=[]
    predicted_test=[]
    residuals_train=[]
    residuals_test=[]
    how_big_test=[]
    how_big_train=[]
   #R2_tracker_train=[]
   #R2_tracker_test=[]
    for i in range(y_tr.shape[1]):
        #i=int(i)
        
        
        tag_name=str('lagged_resuldual')+str(i)
        my_cols=[]
        for k in X_tr.columns:
            my_cols.append(k)
        my_cols.append(tag_name)
    
        X_train_[tag_name]=delta_lagged_train[:,i]
        X_test_[tag_name]=delta_lagged_test[:,i]
    
        # X_data=X_train_.loc[:,str('lagged_resuldual')+str(i)]
        # X_data=np.array(X_data)
        # X_data=X_data.reshape((-1,1))
        # X_data_test=X_test_.loc[:,str('lagged_resuldual')+str(i)]
        # X_data_test=np.array(X_data_test)
        # X_data_test=X_data_test.reshape((-1,1))
        score=make_scorer(mean_absolute_error,greater_is_better=False)
        kernel = ['rbf']
        C = [0.1,1,2,2.5,10,100,200,300,400,500]
        gamma = [10,0.1,300,400,500,600] # find threshold fit best, 
        epsilon = [0.01,0.1]
        shrinking = [True,False]
        svm_grid = {'kernel':kernel,'C':C,'gamma':gamma,'epsilon':epsilon,'shrinking':shrinking}
        SVM = SVR()
        svm_search_hybrid = RandomizedSearchCV(SVM,svm_grid,cv = 2,n_jobs = -1,scoring=score)
       
        
        svm_search_hybrid.fit(X_train_.loc[:,my_cols],delta_train_[:,i])
        #svm_search_hybrid.fit(X_data,delta_train_[:,i])
       
        
        SVRs.append(svm_search_hybrid)
        
        #svr_with_lag=SVR(kernel="rbf", C=2.5, gamma=10, epsilon=0.01)
        #svr_with_lag.fit(X_train_.loc[:,my_cols],delta_train_[:,i])
        #SVRs.append(svr_with_lag)
        
        delta_zero_hat=svm_search_hybrid.predict(X_train_.loc[:,my_cols])
        #delta_zero_hat=svm_search_hybrid.predict(X_data)
        delta_zero_hat_test=svm_search_hybrid.predict(X_test_.loc[:,my_cols]) 
        #delta_zero_hat_test=svm_search_hybrid.predict(X_data_test)
        y_hib_tr=delta_zero_hat+y_pred.iloc[window:,i]
        y_hib_te=delta_zero_hat_test+y_pred_test.iloc[window:,i]
        
        how_big_train.append(delta_train_[:,i]-delta_zero_hat)
        how_big_test.append(delta_test_[:,i]-delta_zero_hat_test)
        predicted_train.append(y_hib_tr)
        predicted_test.append(y_hib_te)
        residuals_train.append(delta_zero_hat)
        residuals_test.append(delta_zero_hat_test)
        if(i%10==0):
            print(str(i)+" th asset fitting done")
    return SVRs,np.array(predicted_train),np.array(predicted_test),np.array(residuals_train),np.array(residuals_test),np.array(how_big_train),np.array(how_big_test)

def SVR_mean_R2_accuracy(predicted,y_data,window):
    accuracy_tracker=[]
    count=0
    for i in range(y_data.shape[1]):
        err=r2_score(y_data.iloc[window:,i],predicted[i])
       # print(err)
        accuracy_tracker.append(err)
        if(err>0.5):
            count=count+1
   
    acc=np.array(accuracy_tracker)
    return acc.mean()
def SVR_mean_MAE_accuracy(predicted,y_data,window):
    accuracy_tracker=[]
    for i in range(y_data.shape[1]):
        accuracy_tracker.append(mean_absolute_error(y_data.iloc[window:,i],predicted[i]))
    acc=np.array(accuracy_tracker)
    return acc.mean()
def SVR_mean_R2_accuracy_test(predicted,y_data,window,start,end):
    accuracy_tracker=[]
    low_performance=[]
    count=0
    for i in range(y_data.shape[1]):
        err=r2_score(y_data.iloc[window+start:end+window,i],predicted[i][start:end])
        accuracy_tracker.append(err)
        print(err)
        
        if(err<0.3):
            low_performance.append(i)
            count=count+1
    print(count)
    print(low_performance)
    acc=np.array(accuracy_tracker)
    return acc.mean()
def SVR_mean_MAE_accuracy_test(predicted,y_data,window,start,end):
    accuracy_tracker=[]
    count=0
    for i in range(y_data.shape[1]):
        err=mean_absolute_error(y_data.iloc[window+start:end+window,i],predicted[i][start:end])

        accuracy_tracker.append(err)
        if(err>0.7):
            count=count+1
    print(count)
    acc=np.array(accuracy_tracker)
    return acc.mean()
def plot_svr_outsample(y_test_normal,predicted_test,window,start,end):
    fig,axes=plt.subplots(4,1,sharex=True)
    
    fig.suptitle('Error-corrected SVR individual stocks')
    axes[0].plot(y_test_normal.iloc[window+start:end+window,0].index,predicted_test[0][start:end],'r-',label=str(reduced_ticker[0]))
    axes[0].plot(y_test_normal.iloc[window+start:end+window,0].index,y_test_normal.iloc[window+start:end+window,0],'g--',label="True Data")
    axes[0].legend(loc=0, prop={'size': 5})

    axes[1].plot(y_test_normal.iloc[window+start:end+window,3].index,predicted_test[3][start:end],'r-',label=str(reduced_ticker[3]))
    axes[1].plot(y_test_normal.iloc[window+start:end+window,3].index,y_test_normal.iloc[window+start:end+window,3],'g--',label="True Data")
    axes[1].legend(loc=0, prop={'size': 5})

    axes[2].plot(y_test_normal.iloc[window+start:end+window,45].index,predicted_test[45][start:end],'r-',label=str(reduced_ticker[45]))
    axes[2].plot(y_test_normal.iloc[window+start:end+window,45].index,y_test_normal.iloc[window+start:end+window,45],'g--',label="True Data")
    axes[2].legend(loc=0, prop={'size': 5})
    
    axes[3].plot(y_test_normal.iloc[window+start:end+window,78].index,predicted_test[78][start:end],'r-',label=str(reduced_ticker[78]))
    axes[3].plot(y_test_normal.iloc[window+start:end+window,78].index,y_test_normal.iloc[window+start:end+window,78],'g--',label="True Data")
    axes[3].legend(loc=0, prop={'size': 5})
    

    plt.xticks(rotation=45)
    plt.show()
def plot_svr_insample(y_train_normal,predicted_train,window,n):
    
    plt.plot(y_train_normal.iloc[window:,n].index,predicted_train[n],'ro',label="Hybrid and with linear regression")
    plt.plot(y_train_normal.iloc[window:,n].index,y_train_normal.iloc[window:,n],'g--',label="True Data")
    plt.xticks(rotation=45)
    plt.title("Residual SVD with Linear Regression")
    plt.legend()
    plt.show()
#hybrid_SVRs,predicted_train,predicted_test,svr_residuals_train,svr_residuals_test=hybrid_SVR_entire(X_train_normal,y_train_normal,
#                                                                                                    X_test_normal,y_test_normal,window=3)


#print("SVR train R2 average: ",SVR_mean_R2_accuracy(predicted_train,y_train_normal,3))
#print("SVR test R2 average: ",SVR_mean_R2_accuracy(predicted_test,y_test_normal,3))

#print("SVR train MAE average: ",SVR_mean_MAE_accuracy(predicted_train,y_train_normal,3))
#print("SVR test MAE average: ",SVR_mean_MAE_accuracy(predicted_test,y_test_normal,3))
#print("SVR test MAE average(bad-time): ",SVR_mean_MAE_accuracy_test(predicted_test,y_test_normal,3,500,610))
#print("SVR test R2 average(Corona_time_period): ",SVR_mean_R2_accuracy_test(predicted_test,y_test_normal,3,,610))
#print("SVR test R2 average(post-corona): ",SVR_mean_R2_accuracy_test(predicted_test,y_test_normal,3,0,500))