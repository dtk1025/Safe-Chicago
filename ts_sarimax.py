#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:48:45 2022

@author: deez
"""

'''
Functions for training time series models
'''

import os
import pandas as pd
import numpy as np
import datetime
import dateutil.parser
import matplotlib.pyplot as plot
import time
from load_data import *
import statsmodels.api as sm
import sklearn.linear_model
import sklearn.metrics
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
import pmdarima as pm
from typing import Union
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler, StandardScaler



def get_data(beat, agg_hours):

    file_path = f'/Users/deez/Documents/WIP/495/Compiled/file upload/crimes_agg_ext_beat{beat}_{agg_hours}h.csv'
    crimes = pd.DataFrame({})

    if(os.path.isfile(file_path)):
        crimes = pd.read_csv(file_path)
    else:
        crimes = get_beat_data(agg_hours, beat)

    return crimes


def prepare_data(data):

    #Drop the columns we don't need
    data = data[['HourGroup','DateTime','BullsGame','CubsGame','SoxGame','BearsGame',
                 'C','m/s','PRCP','SNOW','Count','Holiday']]

    #Standardize the non-boolean columns
    for col in ['C','m/s','PRCP','SNOW']:
        scaler = StandardScaler()
        data[col] = scaler.fit_transform(np.array(data[col]).reshape(-1,1))

    #We will expand our hour group column to become dummies
    data['Month'] = pd.to_datetime(data['DateTime']).dt.month
    data = pd.get_dummies(data, columns=['HourGroup', 'Month']) 
        
    #We scale all the counts
    #count_scaler = MinMaxScaler()
    count_scaler = StandardScaler()
    
    data['Count'] = count_scaler.fit_transform(np.array(data['Count']).reshape(-1,1))

    #We include the target as the next count
    data['Target'] = data['Count'].shift(-1)
    data = data[:-1]
    

    #Filler; nan to zero
    data.fillna(0, inplace=True)
    data.drop(columns='DateTime', inplace=True)
 
    split_index = 4 * (data.shape[0]//5)

    train, test = data.iloc[0:split_index, :],data.iloc[split_index:, :] 

    x_test = test.drop(columns='Target')
    y_test = test['Target']

    x_train = train.drop(columns='Target')
    y_train = train['Target']    
    
    return x_train, x_test, y_train, y_test, count_scaler

def evaluate_model(y_true, y_pred):

    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    mse = sklearn.metrics.mean_squared_error(y_true, y_pred)

    metric_dict = {"R-Squared": r2,
                   "Mean Squared Error": mse}

    return metric_dict


def evaluate_model(y_true, y_pred):

    mse = sklearn.metrics.mean_squared_error(y_true, y_pred, squared=False)
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)

    metric_dict = {"mse": mse,
                   "mae": mae}

    return metric_dict

def SARIMAX_run (beat, agg_hours):  
    #get_data
    crimes = get_data(beat, agg_hours)
    #prep_data
    x_train, x_test, y_train, y_test, count_scaler = prepare_data(crimes)
    #Set up separate validation set
    split_index = 4 * (x_train.shape[0]//5)
    x_val = x_train.iloc[:split_index+1, :]
    x_train = x_train.iloc[split_index+1:, :]
    
    y_val = y_train.iloc[:split_index+1]
    y_train = y_train.iloc[split_index+1:]
    
    length_y = int(round(len(y_train)*0.8,0))
    length_x = int(round(len(x_train)*0.8,0))
    # length_ytest = int(round(len(y_test)*0.2,0))
    
    ##INPUT DATA##
    
    target = y_train
    exog = x_train[['BullsGame', 'CubsGame', 'SoxGame', 'BearsGame', 
      'C', 'm/s', 'PRCP', 'SNOW', 'Holiday']]
    
    p = range(0, 4, 1)
    d = 1
    q = range(0, 4, 1)
    P = range(0, 4, 1)
    D = 0
    Q = range(0, 4, 1)
    s = 4
     
    parameters = product(p, q, P, Q)
    parameters_list = list(parameters)
    
    target_train = target[:length_y]
    exog_train = exog[:length_x]
    
    best_model = SARIMAX(target_train, exog_train, order=(3,1,3), 
      seasonal_order=(0,0,0,4), simple_differencing=False)
    best_model_fit = best_model.fit(disp=False)
    
    n_periods = 6
    y_pred = best_model_fit.predict(n_periods=n_periods, 
                                      exogenous=np.tile(exog_train, 2).reshape(-1,1), 
                                      return_conf_int=True)
    y_test = y_test[:length_y]
    return y_test, y_pred

def run_test(hours, beat_list=None, beat_num=20):
    
    if beat_list == None:
        beat_list = random.sample([i for i in range(1, 78)], beat_num)
        
    performance_mat = []

    for i, beat in enumerate(beat_list):
        #Test LSTM model
        a = SARIMAX_run(beat, hours)
        a_eval = evaluate_model(a[0], a[1])

        #Read in csv to get metrics on count distribution
        a, b, c, y_test, scaler = prepare_data(get_data(beat, hours))
        y_test = np.array(scaler.inverse_transform(pd.DataFrame(y_test))).reshape(y_test.shape[0])

        mean = np.mean(y_test)
        median = np.median(y_test)

        sarimax_mse_pom = a_eval['mse']/mean
        sarimax_mae_pom = a_eval['mae']/mean

        fn = f'crimes_agg_ext_beat{beat}_{hours}h.csv'

        row = [fn, a_eval['mse'], a_eval['mae'], mean]

        performance_mat.append(row)
        
        print(f"Done with beat {beat}...({i+1}/{beat_num})")

    performance_df = pd.DataFrame(performance_mat)
    performance_df.columns =['fn','rmse','mae','y_test_m']


    print(performance_df)
    return performance_df

beat_num = 20
#beat_list = [1,2]
beat_list = [1, 2, 4, 7, 14, 17, 20, 23, 24, 25, 28, 31, 35, 43, 46, 47, 57, 70, 74, 76]
# beat_list = [1]
beat_list.sort()
print(beat_list)
d = run_test(2, beat_list, beat_num)
print("Done with 2 hour.\n")
a = run_test(4, beat_list, beat_num)
print("Done with 4 hour.\n")
b = run_test(6, beat_list, beat_num)
print("Done with 6 hour.\n")
c = run_test(8, beat_list, beat_num)
print("Done with 8 hour.\n")
final = pd.concat([d, a, b, c])
final.reset_index(drop=True, inplace=True)
final.to_csv('sarimax_results.csv') 
    
