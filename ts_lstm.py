'''
Training LSTM time series model and comparing with naive linear baseline
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
import random


import warnings
warnings.filterwarnings("ignore")


def get_data(beat, agg_hours):
    '''
    Gets aggregated data for a specified community area and time segment

    Parameters
    ----------
    beat: community area # (1-77)
    agg_hours: # of hours to keep in each time segment

    Retrrns
    -------
    DataFrame of crime counts and other exogenous features for given CA
    '''

    #Path name of CA
    file_path = f'crimes_agg_ext_beat{beat}_{agg_hours}h.csv'
    crimes = pd.DataFrame({})

    #If it exists, use the file that is already there
    if(os.path.isfile(file_path)):
        crimes = pd.read_csv(file_path)
    #Otherwise, generate using function from load_data.py
    else:
        crimes = get_beat_data(agg_hours, beat)

    return crimes
    

def prepare_data(data):
    '''
    Function to take raw input data and pre-process for model fitting

    Parameters
    ----------
    data: DataFrame for single CA crime counts + features

    Returns
    -------
    -Train and test data split out x / y as well
    -Scaler object to re-scale predicted counts from StandardScale to actual counts
    '''
    
    #Drop the columns we don't need
    data = data[['HourGroup','DateTime','BullsGame','CubsGame','SoxGame','BearsGame',
                 'C','m/s','PRCP','SNOW','Count','Weekday','Holiday']]

    #Standardize the non-boolean columns
    for col in ['C','m/s','PRCP','SNOW']:
        scaler = StandardScaler()
        data[col] = scaler.fit_transform(np.array(data[col]).reshape(-1,1))

    #We will expand our hour group column to become dummies
    data['Month'] = pd.to_datetime(data['DateTime']).dt.month
    data = pd.get_dummies(data, columns=['HourGroup', 'Month','Weekday']) 
        
    #We scale all the counts
    count_scaler = StandardScaler()
    
    data['Count'] = count_scaler.fit_transform(np.array(data['Count']).reshape(-1,1))

    #We include the target as the next count
    data['Target'] = data['Count'].shift(-1)
    data = data[:-1]
    

    #Filler; nan to zero
    data.fillna(0, inplace=True)
    data.drop(columns='DateTime', inplace=True)

    #Setup for train-test split
    split_index = 4 * (data.shape[0]//5)
    train, test = data.iloc[0:split_index, :],data.iloc[split_index:, :] 

    #Split out target values from input data
    x_test = test.drop(columns='Target')
    y_test = test['Target']

    x_train = train.drop(columns='Target')
    y_train = train['Target']    
    
    return x_train, x_test, y_train, y_test, count_scaler

def evaluate_model(y_true, y_pred):
    '''
    Function to generate metrics for evaluation of predictive model performance

    Parameters
    ----------
    y_true: actual counts for test data set
    y_pred: predicted counts from model using test input data

    Returns
    -------
    Dictionary containing model metrics MSE and MAE

    '''
    
    mse = sklearn.metrics.mean_squared_error(y_true, y_pred, squared=False)
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)

    metric_dict = {"mse": mse,
                   "mae": mae}

    return metric_dict


def train_naive(beat, agg_hours):
    ''' 
    Function to train a naive linear model to predict crime counts

    Parameters
    ----------
    beat: community area # (1-77)
    agg_hours: # of hours to keep in each time segment

    Returns
    -------
    y_test: true counts for the test data
    y_pred: predicted counts for the input test data
    coef_dict: dictionary of coefficients for each term in the linear model

    '''

    #Load our data
    crimes = get_data(beat, agg_hours)
        
    #Naive linear model
    lm = sklearn.linear_model.LinearRegression(normalize=True)

    #Use prepare_data to get necessary training inputs 
    x_train, x_test, y_train, y_test, count_scaler = prepare_data(crimes)

    #Fit the model
    lm.fit(x_train, y_train)

    #Make predictions using the trained model
    y_pred = lm.predict(x_test)

    #Recast to numpy arrays after rescaling to counts
    y_pred = np.array(count_scaler.inverse_transform(pd.DataFrame(y_pred))).reshape(y_pred.shape[0])
    y_test = np.array(count_scaler.inverse_transform(pd.DataFrame(y_test))).reshape(y_test.shape[0])

    #Create the coefficient dictionary
    coef_dict = dict(zip(x_train.columns, lm.coef_))

    return y_test, y_pred, coef_dict


def train_lstm(beat, agg_hours):
    ''' 
    Function to train a long short-term memory (LSTM) neural network
    model to predict crime counts

    Parameters
    ----------
    beat: community area # (1-77)
    agg_hours: # of hours to keep in each time segment

    Returns
    -------
    y_test: true counts for the test data
    y_pred: predicted counts for the input test data
    lstm: the trained model itself
    x_test: the input data used to generate y_pred
    '''
    
    crimes = get_data(beat, agg_hours)
    x_train, x_test, y_train, y_test, count_scaler = prepare_data(crimes)

    #Set up separate validation set
    split_index = 4 * (x_train.shape[0]//5)
    x_val = x_train.iloc[:split_index, :]
    x_train = x_train.iloc[split_index:, :]

    y_val = y_train.iloc[:split_index]
    y_train = y_train.iloc[split_index:]

    #Recast to array
    x_train = np.array(x_train).reshape(x_train.shape[0], 1, x_train.shape[1])
    x_val = np.array(x_val).reshape(x_val.shape[0], 1, x_val.shape[1])
    x_test = np.array(x_test).reshape(x_test.shape[0], 1, x_test.shape[1])

    y_train = np.array(y_train).reshape(y_train.shape[0])
    y_val = np.array(y_val).reshape(y_val.shape[0])
    y_test = np.array(y_test).reshape(y_test.shape[0])

    #Define the LSTM model
    lstm = keras.models.Sequential()

    #Add layers
    lstm.add(keras.layers.LSTM(64, activation="relu", return_sequences = True, input_shape=(x_train.shape[1], x_train.shape[2])))
    lstm.add(keras.layers.Dropout(.2))
    lstm.add(keras.layers.LSTM(128, return_sequences = True, activation="relu"))
    lstm.add(keras.layers.Dropout(.2))
    lstm.add(keras.layers.LSTM(256, activation="relu"))
    lstm.add(keras.layers.Dropout(.2))
    lstm.add(keras.layers.Dense(256))
    lstm.add(keras.layers.Dropout(.2))
    lstm.add(keras.layers.Dense(64))
    lstm.add(keras.layers.Dense(1))

    #Compile the model
    lstm.compile(loss='mse', optimizer='adam')

    #Define stopping condition
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)

    #Fit the model 
    lstm.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), callbacks=es, verbose=0)

    #Make predictions using the trained model
    y_pred = np.array(lstm.predict(x_test)).reshape(-1)

    #Recast to arrays after rescaling to counts 
    y_pred = np.array(count_scaler.inverse_transform(pd.DataFrame(y_pred))).reshape(y_pred.shape[0])
    y_test = np.array(count_scaler.inverse_transform(pd.DataFrame(y_test))).reshape(y_test.shape[0])

    return y_test, y_pred, lstm, x_test



def run_test(hours, beat_list=None, beat_num=20):
    '''
    Function to test models trained on random (or defined) list of CAs
    to compare performance across several different inputs

    Parameters
    ----------
    hours: # of hours aggregated together per observation
    beat_list (default None): list of CAs to run the tests over
    beat_num (default 20): number of randomly selected CAs to run the tests over

    Returns
    -------
    DataFrame containing metrics for comparison of performance across CAs

    '''
    #Only use a random # of beats if no list is pre-provided 
    if beat_list == None:
        beat_list = random.sample([i for i in range(1, 78)], beat_num)

    #Define matrix to hold performance metrics 
    performance_mat = []
    
    #Run a test for each beast in the beat_list
    for i, beat in enumerate(beat_list):

        #Create and test LSTM model
        a = train_lstm(beat, hours)
        a_eval = evaluate_model(a[0], a[1])

        #Read in csv to get metrics on count distribution
        a, b, c, y_test, scaler = prepare_data(get_data(beat, hours))
        y_test = np.array(scaler.inverse_transform(pd.DataFrame(y_test))).reshape(y_test.shape[0])

        #Get mean and median of true counts for test set
        mean = np.mean(y_test)
        median = np.median(y_test)

        #Calculate the MSE and MAE as a % of the mean counts for test set
        lstm_mse_pom = a_eval['mse']/mean
        lstm_mae_pom = a_eval['mae']/mean

        #Prepare filename to denote file used as input
        fn = f'crimes_agg_ext_beat{beat}_{hours}h.csv'

        #Compile performance row and add to matrix
        row = [fn, a_eval['mse'], a_eval['mae'], mean]
        performance_mat.append(row)

        #Print progress 
        print(f"Done with beat {beat}...({i+1}/{beat_num})")

    
    #Convert matrix to DataFrame and provide column names
    performance_df = pd.DataFrame(performance_mat)
    performance_df.columns =['fn','rmse','mae','y_test_m']
    
    #Print and return performance DF
    print(performance_df)
    return performance_df

