# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 20:14:19 2021

@author: Beomcheol Kim
"""

from scipy.optimize import minimize
import numpy as np
import pandas as pd
from my_stat_tools import *
##------------------helper functions
# risk budgeting approach optimisation object function
def objective_risk_budget(x, p_cov, rb):
    #risk_measure=np.sqrt(np.dot(np.dot(x,p_cov),x))
    #marginal_risk=np.dot(p_cov,x)/risk_measure
    
    return np.sum((x*np.dot(p_cov, x)/np.dot(x.transpose(), np.dot(p_cov, x))-rb)**2)

def objective_mean_variance(x,p_cov,p_mu,risk_aversion=5):
    return (1.0/2.0)*np.dot(x.transpose(), np.dot(p_cov, x))-risk_aversion*np.dot(x.transpose(),p_mu)

# constraint on sum of weights equal to one
def constraint_sum_weight(x):
    return np.sum(x)-1.0


# constraint on weight larger than zero
def constraint_long_only_weight(x):
    return x
##---------------------------------------------------
## optimizers
def rb_p_weights(covariance, rb):
# number of FTSE 100 index components 
    num_sub_ftse = covariance.shape[1]
    # covariance matrix of asset returns
    p_cov = covariance
    # initial weights-equal weight portfolio
    w0 = 1.0 * np.ones((num_sub_ftse, 1)) / num_sub_ftse
    # constraints
    #passing function directly
    cons = ({'type': 'eq', 'fun': constraint_sum_weight}, {'type': 'ineq', 'fun': constraint_long_only_weight})
    # portfolio optimisation
    return minimize(objective_risk_budget, w0, args=(p_cov, rb), method='SLSQP', constraints=cons)

def mv_p_weights(covariance, mean):
# number of FTSE 100 index components 
    num_sub_ftse = covariance.shape[1]
    # covariance matrix of asset returns
    p_cov = covariance
    # initial weights-equal weight portfolio
    w0 = 1.0 * np.ones((num_sub_ftse, 1)) / num_sub_ftse
    # constraints
    #passing function directly
    cons = ({'type': 'eq', 'fun': constraint_sum_weight}, {'type': 'ineq', 'fun': constraint_long_only_weight})
    # portfolio optimisation
    return minimize(objective_mean_variance, w0, args=(p_cov, mean), method='SLSQP', constraints=cons)

def factor_signal_weight(sensitivity_matrix,factor_name,tickers):
    # create a dataframe with the factor loading matrix
    df=pd.DataFrame(sensitivity_matrix,index=tickers)
    
    # substitute 0.0 for nan and collect only positive sensitivities
    df=df[df>0].fillna(0.0)
    if(factor_name=='size'): 
        
        weight=df.iloc[:,0]/np.sum(df.iloc[:,0])
        #print(weight)
    elif(factor_name=='value'):
        weight=df.iloc[:,1]/np.sum(df.iloc[:,1])
    elif(factor_name=='momentum'):
        weight=df.iloc[:,2]/np.sum(df.iloc[:,2])
    else:
        print("error, invalid factor name")
    return weight
def error_signal_weight(support_residual,tickers):
    # from the prediction produced by the support vector regression, create a data frame
    df=pd.DataFrame(support_residual,index=tickers)
    df=df.fillna(0.0)
    df=df[df>=0] # consider only positive errors
    weight=np.abs(df.mean(axis=1))/np.sum(np.abs(df.mean(axis=1)))
    return weight
def Gini_index(weight_vector):
    
    n=weight_vector.shape[0]
    weight=np.sort(weight_vector)
    
    monotone_inc_vector=np.arange(1,n+1,1)
    denominator=n*np.sum(weight)
    nominator=2.0*np.sum(monotone_inc_vector*weight)
    
    return nominator/denominator-((n+1)/n)
    
    