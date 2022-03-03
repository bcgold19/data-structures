# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 20:16:36 2021

@author: Beomcheol Kim
"""
import numpy as np
import pandas as pd
from my_stat_tools import *
from weight_optimizers import *
from scipy.optimize import minimize
from sklearn.metrics import make_scorer
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV



def smart_beta_regression(target,window,new_ex_return,new_factor,start,end='2021-11-19'):
    # reducing target time-line
    flag_end=new_ex_return.index<=end
    flag_start=new_ex_return.index>=start
    n=new_ex_return.shape[1]
    p_dates=new_ex_return.index[flag_start & flag_end]
    
    # standardising only x-data
    new_factor_norm,_,_=standardised(new_factor)
    
    # initialise portfolio weights matrix
    weight_matrix = pd.DataFrame(index=p_dates, columns=new_ex_return.columns)
    portfolio_rets = pd.DataFrame(index=p_dates, columns=[target])
    portfolio_reports = pd.DataFrame(index=p_dates, columns=['Expected Return','Variance'])
    
    risk_contribution_rb=pd.DataFrame(index=p_dates,columns=new_ex_return.columns)
    
    
    #initialize rebalancing date
    rebalancing_date=p_dates[0]
    
    for i in range(len(p_dates)):
        if(p_dates[i]==rebalancing_date):
            ## window-periods look back estimations of regression
            target_end_date=p_dates[i]
            target_start_date=p_dates[i]-np.timedelta64(window,'D')
            target1=new_ex_return.index<target_end_date
            target2=new_ex_return.index>=target_start_date
            
            # calling a regression estimation
            estimators=EMalgorithm(new_factor_norm[target1 & target2],new_ex_return[target1 & target2])
            mu_f=estimators[0]
            sigma_f=estimators[1]
            B=estimators[2]
            alpha=estimators[3]
            Psi=estimators[4]
            ret_cov=B@sigma_f@B.T+Psi
            p_mu=alpha+B@mu_f
            n=ret_cov.shape[1]
            
            # Based on the regression result, calculate weights
            if(target=='mv'):
                weight_matrix.loc[p_dates[i]] = mv_p_weights(ret_cov, p_mu).x
                #w_t=weight_matrix.loc[p_dates[i]]
        
            elif(target=='rb'):
                weight_matrix.loc[p_dates[i]]=rb_p_weights(ret_cov, 1.0/n).x
                #w_t=weight_matrix.loc[p_dates[i]]
            else:
                print('Invalid_target')
                
            
         
            # Based on the weight, calculate key statistics for portfolio
            w_t=weight_matrix.loc[p_dates[i]]
            b_t=np.array([1]*n)*(1.0/n) ## benchmark as an equally weighted portfolio
            diffs=w_t-b_t
            risk_contribution_rb.loc[p_dates[i]]=np.dot(ret_cov,w_t)/np.dot(w_t.transpose(), np.dot(ret_cov, w_t))
            portfolio_reports.loc[p_dates[i],'Expected Return']=np.dot(p_mu,w_t)
            portfolio_reports.loc[p_dates[i],'annualised_Exp_Return']=np.dot(p_mu,w_t)*window
            portfolio_reports.loc[p_dates[i],'benchmark_return']=np.dot(p_mu,b_t)*window
            portfolio_reports.loc[p_dates[i],'diff_vols']=np.sqrt(np.dot(diffs.transpose(), np.dot(ret_cov, diffs))*window)
            portfolio_reports.loc[p_dates[i],'Variance']=np.dot(w_t.transpose(), np.dot(ret_cov, w_t))
            portfolio_reports.loc[p_dates[i],'annualised_Vol']=np.sqrt(np.dot(w_t.transpose(), np.dot(ret_cov, w_t))*window)
            portfolio_reports.loc[p_dates[i],'Sharpe Ratio']=np.dot(p_mu,w_t)*window/np.sqrt(np.dot(w_t.transpose(), np.dot(ret_cov, w_t))*window)
            portfolio_reports.loc[p_dates[i],'Treynor Ratio']=np.dot(p_mu,w_t)*window/np.sqrt(np.dot(w_t.transpose(), B[:,2])*window)
            portfolio_reports.loc[p_dates[i],'Information Ratio']=np.dot(p_mu,diffs)*window/np.sqrt(np.dot(diffs.transpose(), np.dot(ret_cov, diffs))*window)
            portfolio_reports.loc[p_dates[i],'Herfindahl']=(n*np.dot(w_t,w_t)-1.0)/(n-1)
            portfolio_reports.loc[p_dates[i],'Gini']=Gini_index(w_t)
            
            rebalancing_date=rebalancing_date+np.timedelta64(window, 'D')
        
        else:
            weight_matrix.loc[p_dates[i]] = weight_matrix.iloc[weight_matrix.index.get_loc(p_dates[i])-1]
            risk_contribution_rb.loc[p_dates[i]]=np.dot(ret_cov,w_t)/np.dot(w_t.transpose(), np.dot(ret_cov, w_t))
        portfolio_rets.loc[p_dates[i]] = np.sum(weight_matrix.loc[p_dates[i]] * new_ex_return.loc[p_dates[i]]) 
    print(p_dates[0],p_dates[-1])
    return weight_matrix,portfolio_rets,portfolio_reports.dropna()

def equal_weight_port(new_ex_return,start,end):
    # creating an equal weight portfolio.
    n=new_ex_return.shape[1]
    ew_return=pd.DataFrame(np.sum(1.0*new_ex_return[(new_ex_return.index>=start) & (new_ex_return.index<=end)]/n, axis=1), columns=['Equal Weighted'])
  
    return ew_return


#mv_weight,mv_return,mv_report=smart_beta_regression('mv', 91, ex_rets, new_factor, start)
def ftse_performance(factor_matrix,start,end):
    flag_end=factor_matrix.index<=end
    flag_start=factor_matrix.index>=start
    
    p_dates=factor_matrix.index[flag_start & flag_end]
    ftse=factor_matrix.loc[p_dates,'MKT']
    return ftse

# my_equals=equal_weight_port(ex_rets, start, end)
# rb_cumsums=rb_return.cumsum()
# mv_cumsums=mv_return.cumsum()
# size_cumsums=size_return.cumsum()
# value_cumsums=value_return.cumsum()
# mom_cumsums=mom_return.cumsum()
# svr_mv_sumsums=svr_mvs_return.cumsum()
# svr_signal_cumsums=svr_err_return.cumsum()
# ew_cumsums=my_equals.cumsum()
# ftses=ftse_performance(factors,start,end)
# ftse_cumsums=ftses.cumsum()
# svrrb_cumsums=svr_rbs_return.cumsum()
# pd.concat([rb_cumsums,ew_cumsums,ftse_cumsums,size_cumsums,svr_signal_cumsums],axis=1).iloc[30:,:].plot()
# plt.title('The Performances of a few portfolios')

# rb_sharpe.plot(label='ERP')
# rbs_sharpe.plot(label='Support Vector ERP')
# plt.title('Information Ratio Comparison: Benchmark- Equal weight')
# plt.legend()
# plt.show()
def reporting_target(target,reporting_list):
    df=pd.DataFrame(index=reporting_list[0].index)
    name=['_mv','_rb','_size','_value','_mom','_ECSVR_RB','_svr_signal']
    for i in range(len(reporting_list)):
        df[target+name[i]]=(reporting_list[i])[target]
        #print(df)
    return df      
def ftse_performance(factor_matrix,start,end):
    flag_end=factor_matrix.index<=end
    flag_start=factor_matrix.index>=start
    
    p_dates=factor_matrix.index[flag_start & flag_end]
    ftse=factor_matrix.loc[p_dates,'MKT']
    return ftse
def plot_scatter_returns(data1,data2,context):
    plt.scatter(data1,data2)
    plt.title(context)