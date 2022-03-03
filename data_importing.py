# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:45:24 2021

@author: Beomcheol Kim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def import_data_set():
    '''
    

    Returns
    -------
    my_tickers : TYPE : list
        DESCRIPTION. : name of all tickers in the ftse 100
    df_rf : TYPE pandas series
        DESCRIPTION. time series of risk free rate -SONIA benchmark
    df1 : TYPE :  data frame 
        DESCRIPTION. : log return time series for each assets
    df2 : Type : data frame
        DESCRIPTION. book to ratio
    df3 : TYPE
        DESCRIPTION. current market cap

    '''

    # Importing data set
    ftse_daily=pd.read_excel('Book2.xlsx',index_col=0,header=[0,2])
    ftse_data=ftse_daily
    ftse_data.index=pd.to_datetime(ftse_data.index)
    
    #collect tickers
    tickers=ftse_data.columns.to_list()
    my_ticker=[]
    for i in range(len(tickers)):
            if(tickers[i][0]=='UKX Index' or tickers[i][0]=='SONIO/N Index'):
                continue
            else:
                my_ticker.append(tickers[i][0])
        
    temp=np.unique(my_ticker)
    my_tickers=list(temp)
    
    testing=ftse_data
    ## data preparation
    df_rf=ftse_data.loc[:,('SONIO/N Index','PX_LAST')]
    df_rf=df_rf*0.01
    ## calculating log return
    df1= np.log(testing.loc[:,(my_tickers,'PX_LAST')]) - np.log(testing.loc[:,(my_tickers,'PX_LAST')].shift(1))
    df2=testing.loc[:,(my_tickers,'PX_TO_BOOK_RATIO')] ## price book to ratio
    df3=testing.loc[:,(my_tickers,'CUR_MKT_CAP')] ## historical market cap
    mkt=(np.log(ftse_data.loc[:,('UKX Index','PX_LAST')])-np.log(ftse_data.loc[:,('UKX Index','PX_LAST')].shift(1)))-df_rf
    mkt.index=pd.to_datetime(mkt.index)
    return my_tickers,df_rf,df1,df2,df3,mkt



def get_factors():
    '''
    

    Returns
    -------
    risk_free : Pandas Series
        DESCRIPTION.
        - The sequence of risk free rate(SONIA)
    returns : TYPE : Pandas DataFrame
        DESCRIPTION.
        - return of all stocks with time periods
    new_ex_return : TYPE : Pandas DataFrame
        DESCRIPTION.
        - excess return, return - risk_free, except the data contains nan
    new_factor : TYPE Pandas DataFrame
        DESCRIPTION.
        - Factors, sharing time period with the new_ex_return

    '''
    #importing data from the previous function
    tickers,risk_free,returns,bp_ratio,mkt_cap,mkt=import_data_set()
    
    #calculating excess return by subtracting risk free rate for each time period and each stocks
    exx_return=returns.copy()
    for i in range(exx_return.shape[1]):
        exx_return.iloc[:,i]=returns.iloc[:,i]-risk_free
    n=returns.shape[1]
    
    ## setting a ratio
    loc10=math.floor(n/10)
    loc20=loc10*2
    loc30=loc10*3
    loc40=loc10*4
    loc50=loc10*5
    loc60=loc10*6
    loc70=loc10*7
    loc80=loc10*8
    loc90=loc10*9
    
    smb_list=[]
    hml_list=[]
    sh_list=[]
    sm_list=[]
    sl_list=[]
    bh_list=[]
    bm_list=[]
    bl_list=[]
    
    # Calculating factors 
    for i in range(returns.shape[0]):
        
        # sorting them
        temp=bp_ratio.iloc[i,:].sort_values()
        temp1=mkt_cap.iloc[i,:].sort_values()

        
        growth_temp=temp.index[0:loc30] # 0~ 30th quantile 
        neutral_temp=temp.index[loc30:loc70] # 30~70th quantile
        value_temp=temp.index[loc70:n] # 70 to the last quantile

        small_cap=temp1.index[0:loc40] # upto 40th quantile
        big_cap=temp1.index[loc60:n] # 60th to the last

        # create lists of tickers for each group
        growth_ticker=[x[0] for x in growth_temp]  
        neutral_ticker=[x[0] for x in neutral_temp]
        value_ticker=[x[0] for x in value_temp]

        small_ticker=[x[0] for x in small_cap]
        big_ticker=[x[0] for x in big_cap]

        # create ticker sets by intersection
        sh_ticker=list(set(small_ticker)&set(value_ticker))
        sm_ticker=list(set(small_ticker)&set(neutral_ticker))
        sl_ticker=list(set(small_ticker)&set(growth_ticker))


        bh_ticker=list(set(big_ticker)&set(value_ticker))
        bm_ticker=list(set(big_ticker)&set(neutral_ticker))
        bl_ticker=list(set(big_ticker)&set(growth_ticker))

        # average returns of each group
        sh=returns.loc[:,(sh_ticker,'PX_LAST')].iloc[i].mean()
        sm=returns.loc[:,(sm_ticker,'PX_LAST')].iloc[i].mean()
        sl=returns.loc[:,(sl_ticker,'PX_LAST')].iloc[i].mean()
        bh=returns.loc[:,(bh_ticker,'PX_LAST')].iloc[i].mean()
        bm=returns.loc[:,(bm_ticker,'PX_LAST')].iloc[i].mean()
        bl=returns.loc[:,(bl_ticker,'PX_LAST')].iloc[i].mean()
    
        sh_list.append(sh)
        sm_list.append(sm)
        sl_list.append(sl)
        bh_list.append(bh)
        bm_list.append(bm)
        bl_list.append(bl)

        # calculation of the factor 
        smb=(sh+sm+sl)/3-(bh+bm+bl)/3
        hml=(bh+sh)/2-(bl+sl)/2
        smb_list.append(smb)
        hml_list.append(hml)

        
        # 
    mkt_list=list(mkt)
    sh_arr=np.array(sh_list)
    sm_arr=np.array(sm_list)
    sl_arr=np.array(sl_list)
    bh_arr=np.array(bh_list)
    bm_arr=np.array(bm_list)
    bl_arr=np.array(bl_list)
    dff=pd.DataFrame({'sh':sh_arr,'sm':sm_arr,'sl':sl_arr,
                           'bh':bh_arr,'bm':bm_arr,'bl':bl_arr
                          },index=returns.index)
    

    
    # in order to calculate the momentum
    
    smb_arr=(dff['sh']+dff['sm']+dff['sl'])/3-(dff['bh']+dff['bm']+dff['bl'])/3
    hml_arr=(dff['bh']+dff['sh'])/2-(dff['bl']+dff['sl'])/2
    mom_df=pd.DataFrame({'sh':dff['sh'],
                    'bh':dff['bh'],
                    'sl':dff['sl'],
                    'bl':dff['bl']},index=mkt.index)
    new_mom=0.5*(mom_df['sh'].rolling(252).mean()+mom_df['bh'].rolling(252).mean()-mom_df['sh'].rolling(21).mean()-mom_df['bh'].rolling(21).mean()) \
            -0.5*(mom_df['sl'].rolling(252).mean()+mom_df['bl'].rolling(252).mean()-mom_df['sl'].rolling(21).mean()-mom_df['bl'].rolling(21).mean())
    factor_df = pd.DataFrame(
    {'SMB': smb_arr,
     'HML': hml_arr,
     'MKT': mkt_list,
     'MOM': new_mom
    })
    factor_df.index=returns.index
    exx_return=exx_return[1:]
    factor_df=factor_df[1:]
    # as a consequence of the mom calculation, starting date to the 501th data for mom is missing
    # hence considering only valid dataset
    new_ex_return=exx_return.iloc[502:,:]
    new_factor=factor_df.iloc[502:,:]
    # removing columns that contains nothing
    new_ex_return.dropna(axis=1, how="any", thresh=None, subset=None, inplace=True)
    return risk_free,returns,new_ex_return,new_factor
    
        
def collect_tickers(ret_data):
    
    tickers=ret_data.columns.to_list()
    reduced_ticker=[]
    for i in range(len(tickers)):
        if(tickers[i][0]=='UKX Index' or tickers[i][0]=='SONIO/N Index'):
            continue
        else:
            reduced_ticker.append(tickers[i][0])
    return reduced_ticker



def performance_factor_plots(factor_data,targets,names):
    '''
    Parameters
    ----------
    factor_data : TYPE :DataFrame
        DESCRIPTION. naive data for factors
    targets : TYPE list of string
        DESCRIPTION. 'MOM','HML','SMB','MKT' are possible choices
    tittle : Name of title for your plot string
    Returns
    -------
    None.

    '''
    factor_data[targets].cumsum().plot()
    plt.title(names)
    plt.legend()
    plt.show()    
    



def correlation_bad(y_train,X_train,targets):
    tracker=[]
    for i in range(len(targets)):
        corrs=np.corrcoef(y_train.iloc[:,targets[i]],X_train['SMB'])[0][1]
        tracker.append(corrs)
    return np.array(tracker).mean()
        
        
def not_bad_corr(y_train,X_train,targets):
    tracker=[]
    for i in range(y_train.shape[1]):
        if(i not in targets):
            corrs=np.corrcoef(y_train.iloc[:,i],X_train['SMB'])[0][1]
            tracker.append(corrs)
    return np.array(tracker).mean()
        
        
        
        
        
        
        
        
        
        
        
    




