# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 20:19:02 2021

@author: Beomcheol Kim
"""
import numpy as np
import pandas as pd
from my_stat_tools import *
from weight_optimizers import *
def factor_signal_portfolio(factor_name,window,new_ex_return,new_factor,tickers,start,end='2021-11-19'):
    flag_end=new_ex_return.index<=end
    flag_start=new_ex_return.index>=start
    p_dates=new_ex_return.index[flag_start & flag_end]

    # initialise portfolio weights matrix
    weight_matrix_factor = pd.DataFrame(index=p_dates, columns=tickers)
    # initialise portfolio return matrix
    portfolio_rets_factor = pd.DataFrame(index=p_dates, columns=[factor_name])
    portfolio_reports_factor = pd.DataFrame(index=p_dates, columns=['Expected Return','Variance'])
    

    rebalancing_date_factor=p_dates[0]
    for i in range(len(p_dates)):
        if(p_dates[i]==rebalancing_date_factor):
            ## window-periods look back estimations of regression
            target_end_date=p_dates[i]
            target_start_date=p_dates[i]-np.timedelta64(window,'D')
            target1=new_ex_return.index<target_end_date
            target2=new_ex_return.index>=target_start_date
            
            estimators=EMalgorithm(new_factor[target1 & target2],new_ex_return[target1 & target2])
            mu_f=estimators[0]
            sigma_f=estimators[1]
            B=estimators[2]
            alpha=estimators[3]
            Psi=estimators[4]
            ret_cov=B@sigma_f@B.T+Psi
            p_mu=alpha+B@mu_f
            n=ret_cov.shape[1]
            weight_matrix_factor.loc[p_dates[i]] = factor_signal_weight(B,factor_name,tickers)
            # updating rebalancing date
            #print(weight_matrix_factor)
            rebalancing_date_factor=rebalancing_date_factor+np.timedelta64(window, 'D')
        
            w_t=weight_matrix_factor.loc[p_dates[i]]
            b_t=np.array([1]*88)*(1.0/88.0) ## benchmark as an equally weighted portfolio
            diffs=w_t-b_t
           # risk_contribution_rb.loc[p_dates[i]]=np.dot(ret_cov,w_t)/np.dot(w_t.transpose(), np.dot(ret_cov, w_t))
            portfolio_reports_factor.loc[p_dates[i],'Expected Return']=np.dot(p_mu,w_t)
            portfolio_reports_factor.loc[p_dates[i],'annualised_Exp_Return']=np.dot(p_mu,w_t)*91.0
            portfolio_reports_factor.loc[p_dates[i],'Variance']=np.dot(w_t.transpose(), np.dot(ret_cov, w_t))
            portfolio_reports_factor.loc[p_dates[i],'benchmark_return']=np.dot(p_mu,b_t)*window
            portfolio_reports_factor.loc[p_dates[i],'diff_vols']=np.sqrt(np.dot(diffs.transpose(), np.dot(ret_cov, diffs))*window)
            portfolio_reports_factor.loc[p_dates[i],'annualised_Vol']=np.sqrt(np.dot(w_t.transpose(), np.dot(ret_cov, w_t))*91.0)
            portfolio_reports_factor.loc[p_dates[i],'Sharpe Ratio']=np.dot(p_mu,w_t)*91.0/np.sqrt(np.dot(w_t.transpose(), np.dot(ret_cov, w_t))*91.0)
            portfolio_reports_factor.loc[p_dates[i],'Treynor Ratio']=np.dot(p_mu,w_t)*91.0/np.sqrt(np.dot(w_t.transpose(), B[:,2])*91.0)
            portfolio_reports_factor.loc[p_dates[i],'Information Ratio']=np.dot(p_mu,diffs)*91.0/np.sqrt(np.dot(diffs.transpose(), np.dot(ret_cov, diffs))*91.0)
            portfolio_reports_factor.loc[p_dates[i],'Herfindahl']=(88.0*np.dot(w_t,w_t)-1.0)/(87.0)
            portfolio_reports_factor.loc[p_dates[i],'Gini']=Gini_index(w_t)
        else:
            weight_matrix_factor.loc[p_dates[i]] = weight_matrix_factor.iloc[weight_matrix_factor.index.get_loc(p_dates[i])-1]
        
        portfolio_rets_factor.loc[p_dates[i]] = np.sum(weight_matrix_factor.loc[p_dates[i]] * new_ex_return.loc[p_dates[i]]) 
        
    return weight_matrix_factor,portfolio_rets_factor,portfolio_reports_factor.dropna()