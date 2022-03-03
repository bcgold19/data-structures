# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:21:33 2021

@author: victo
"""

import numpy as np
import data_importing as data
import my_stat_tools as stt
import numexpr as ne

def rbf_kernel_matrix(X,gamma=0.01,var=5.0):

    X_norm = np.einsum('ij,ij->i',X_train_normal,X_train_normal)
    K = ne.evaluate('v * exp(-g * (A + B - 2 * C))', {
        'A' : X_norm[:,None],
        'B' : X_norm[None,:],
        'C' : np.dot(X_train_normal, X_train_normal.T),
        'g' : gamma,
        'v' : var
    })
    return K

def fit_LSSVR(X,y,C):
    # will return optimal alphas and b
    # X is in R(n_sample,n_feature)
    
    n_samples,n_features=X.shape
    ker_matrix=rbf_kernel_matrix(X)
    
    ## calculate K+(1/C)*identity
    ones=np.ones((1,X_train_normal.shape[0]))
    ones_a=np.ones((1+X_train_normal.shape[0],1))
    K=ker_matrix+(1/C)*np.identity(ker_matrix.shape[0])
    K_=np.append(ones_a,np.vstack([ones,K]),1)
    
    
    y_i=[0]+list(y)
    y_i=np.array(y_i)
    
    coeffs=np.linalg.inv(K_)@y_i
    return coeffs[0],coeffs[1:]