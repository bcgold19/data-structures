# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 12:27:34 2021

@author: victo
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *


def p_distribution_graphs(factor_data):
    fig,axes=plt.subplots(2,2,sharex=True)
    axes[0,0].set_title('Size Distribution')
    axes[0,1].set_title('Value Distribution')
    axes[1,0].set_title('Market Excess Return Distribution')
    axes[1,1].set_title('Momentum Distribution')
    sns.distplot(factor_data['SMB'],ax=axes[0,0])

    sns.distplot(factor_data['HML'],rug=True,ax=axes[0,1])
    sns.distplot(factor_data['MKT'],rug=True,ax=axes[1,0])
    sns.distplot(factor_data['MOM'],rug=True,ax=axes[1,1])
    
    
    plt.show()
    

#sns.boxplot(data=factors)
#plt.title('Bar-plots of factors')
#plt.show()

def daily_performance_four(y_data,tickers):
    plt.plot(y_data.index,y_data.iloc[:,0],label=str(tickers[0]))
    plt.plot(y_data.index,y_data.iloc[:,21],label=str(tickers[21]))
    plt.plot(y_data.index,y_data.iloc[:,58],label=str(tickers[58]))
    plt.plot(y_data.index,y_data.iloc[:,85],label=str(tickers[85]))
    plt.legend()
    plt.xticks(rotation=45)
    plt.title("Four stock's daily performances")
    plt.show()
    
    
def rolling_sum_of_four(y_data,tickers,window):
    plt.plot(y_data.index,y_data.iloc[:,0].rolling(window).sum(),label=str(tickers[0]))
    plt.plot(y_data.index,y_data.iloc[:,21].rolling(window).sum(),label=str(tickers[21]))
    plt.plot(y_data.index,y_data.iloc[:,58].rolling(window).sum(),label=str(tickers[58]))
    plt.plot(y_data.index,y_data.iloc[:,85].rolling(window).sum(),label=str(tickers[85]))
    plt.legend()
    plt.xticks(rotation=45)
    plt.title("Four stock's performances"+str(window)+"holding")
    plt.show()    
    
from statsmodels.graphics.tsaplots import plot_acf
    
##log-return of FTSE 100 index contains alomost no auto correlation. That implies
# no useful pattern on the time series data
plot_acf(factors['MKT'],lags=252)
plt.title("FTSE 100 index Auto Correlation")
plt.show()

    
    
    
    
    
    
    
    