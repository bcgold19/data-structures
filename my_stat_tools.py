# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:52:52 2021

@author: Beomcheol Kim
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
## Since the FTSE 100 index historically showed a significant underperformance, it would be better to choose the assets with low sens
## -tivity with MKT factor. 


def standardised(data):
    '''
    

    Parameters
    ----------
    data : training set and test set of the factors

    Returns
    -------
    TYPE   Pandas DataFrame
        DESCRIPTION.
        - result of standardising
    mean : TYPE numpy array
        DESCRIPTION.
        - time varying mean of each stocks
    std : TYPE 
        DESCRIPTION.

    '''
    mean=np.mean(data,axis=0)
    std=np.std(data,axis=0)
    return (data-mean)/std,mean,std
def re_standardised(data,mean,std):
    return data*std+mean


def train_test_split(data,factors,ratio):
    ## input : 
    #    data : (T x n_assets )
    #    factors : (T x n_factors)
    #    Ratio : the ratio of training set normally ratio is 0.7 of the entire data set
    
    ## Outputs: Splited data sets 
    
    
    if(data.shape[0]==factors.shape[0]):
        threshold=int(data.shape[0]*ratio)
        y_train=data[:threshold]
        y_test=data[threshold:]
        X_train=factors[:threshold]
        X_test=factors[threshold:]
    else:
        print("Check dimensionalities of your input data")
        print("Data dim : ", data.shape)
        print("Factor dim: ", factors.shape)
    return [X_train,y_train,X_test,y_test]

## Gaussian Assumption with ML estimation : by EM algorithm
def EMalgorithm(X,y):
    # input  
    #   X: factor data ( T x number of factors)- dimension
    #   y: return data  (T x number of assets) -dimension
    
    # return : estimator of each parameters
    #  mu_fa : estimated mean for the factor ( n_factor x 1 ) - vector
    #  sigma_fa : estimated covariance matrix for factors ( n_factor x n_factor ) -matrix
    #  B : estimated factor loading matrix  (number_of_assets x number_of factors) -matrix
    # alpha : estimated alpha 
    # Psi : estimated covariance for the residual term - Assumed that each residual are independent to factor and other residuals
    F=X.T
    T=X.shape[0]
    y_bar=(np.sum(y,axis=0))/T
    
    mu_fa=np.sum(F,axis=1)/T
    
    temp=F-np.outer(mu_fa,np.ones((1,T)))
    sigma_fa=(temp@temp.T)/T
    first=T*np.outer(y_bar,mu_fa)-np.dot(y.T,X)
    second=(T*np.outer(mu_fa,mu_fa)-np.dot(X.T,X))
    B=first@np.linalg.inv(second)
    alpha=y_bar-np.dot(B,mu_fa)
    temp3=(y.T-np.outer(alpha,np.ones((1,T)))-np.dot(B,X.T))
    Psi=(temp3@temp3.T)/T
    d_elements=np.diagonal(Psi)
    Psi=np.diag(d_elements)
    return [mu_fa,sigma_fa,B,alpha,Psi]

def reported_betas(X,y,estimators,tickers):

    B=estimators[2]
    Betas=pd.DataFrame(B,index=tickers,columns=X.columns)
    Betas.index.name='tickers'
    Betas['name']=Betas.index
    
    return Betas 
    
def average_outsample_accuracy(y,y_pred):
    '''
    

    Parameters
    ----------
    y : TYPE Numpy array
        DESCRIPTION.
        - Actual excess returns
    y_pred : TYPE Numpy array
        DESCRIPTION.
        - the result of regression, predected

    Returns
    -------
    TYPE numeric
        DESCRIPTION.
        - Average of the R_squared 

    '''
    accuracy_tracker=[]
    for i in range(y_pred.shape[1]):
        accuracy_tracker.append(r2_score(y.iloc[:,i],y_pred.iloc[:,i]))
        
        print(r2_score(y.iloc[:,i],y_pred.iloc[:,i]))
    acc=np.array(accuracy_tracker)
    return acc.mean()

def reg_mean_MAE_accuracy(y,y_pred):
    accuracy_tracker=[]
    for i in range(y_pred.shape[1]):
        
        accuracy_tracker.append(mean_absolute_error(y.iloc[:,i],y_pred.iloc[:,i]))
    acc=np.array(accuracy_tracker)
    return acc.mean()
def average_outsample_acc(y,y_pred,start,end):
    accuracy_tracker=[]
    count=0
    for i in range(y.shape[1]):
        err=r2_score(y.iloc[start:end,i],y_pred.iloc[start:end,i])
        print(err)
        accuracy_tracker.append(err)
        if(err>0.5):
            count=count+1
    print(count)
    acc=np.array(accuracy_tracker)
    return acc.mean()
def average_outsample_MAE(y,y_pred,start,end):
    accuracy_tracker=[]
    count=0
    for i in range(y.shape[1]):
        err=mean_absolute_error(y.iloc[start:end,i],y_pred.iloc[start:end,i])
        
        accuracy_tracker.append(err)
        if(err>0.5):
            count=count+1
    print(count)
    acc=np.array(accuracy_tracker)
    return acc.mean()    
    
    
#sort_data_MKT=df_beta[df_beta["MKT"]<0.90].sort_values(by='MKT',ascending=True)
#sort_data_HML=df_beta[df_beta["HML"]>-0.015397].sort_values(by='HML',ascending=False)
#sort_data_SMB=df_beta[df_beta["SMB"]>0.037275].sort_values(by='SMB',ascending=False)
#sort_data_MOM=df_beta[df_beta["MOM"]>0.037275].sort_values(by='MOM',ascending=False)
    
    
#b=sns.barplot(data=sort_data_MKT[:20],x='MKT',y='name',palette='Reds_r')
#b.axes.set_title("Assets with lower sensitivities to the FTSE 100 index",fontsize=10)
#plt.figure(figsize=(100,1))
#plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
