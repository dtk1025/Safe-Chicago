'''
Functions for generating model input data for future predictions
'''


import os
import pandas as pd
import numpy as np
import datetime
import dateutil.parser
import matplotlib.pyplot as plot
import time
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

#########
#Weather#
#########

def generate_future_weather(hour_delta=1):
    '''
    Function to impute future weather data for model input based on historical data

    Parameters
    ----------
    hour_delta (default 1): the difference in hours between each observation

    Returns
    -------
    DataFrame containing imputed weather metrics from the end of historical data
    until the end of 2023
    '''

    #Look for clean historical weather data, otherwise generate it
    file_path = f'historical_weather_clean_{hour_delta}h.csv'

    if(os.path.isfile(file_path)):
        weather = pd.read_csv(file_path)
    else:
        weather = compile_weather(file_name, hour_delta)

    #We drop the relative humidity feature as it is unavailable for a large portion of the dataset
    weather.drop(columns='RelHum', inplace=True)
    
    #Sort it
    weather.sort_values(by=['Year','Month','Day','HourGroup'], inplace=True)

    #Get our starting point to generate data from
    last_year = str(weather.iloc[-1,1])
    
    last_month = str(weather.iloc[-1,4])
    
    #zero pad year if necessary
    if(len(last_month) == 1):
        last_month = "0"+last_month

    last_day = str(weather.iloc[-1,5])

    #zero pad day if necessary
    if(len(last_day) == 1):
        last_day = "0"+last_day

    last_hour = str(weather.iloc[-1,6])
    if(len(last_hour) == 1):
        last_day = "0"+last_hour

    #String format the last date so we can convert to datetime object
    last_weather_string = f'{last_month}/{last_day}/{last_year} {last_hour}:00:00'
    last_datetime = datetime.datetime.strptime(last_weather_string, "%m/%d/%Y %H:%M:%S")
    
    year = int(last_year)
    date = last_datetime + datetime.timedelta(hours = hour_delta)

    #New DF to store data
    future_weather = pd.DataFrame({})

    #Generate until we've hit the end of 2023
    while int(year) < 2024:

        year = int(date.year)
        month = int(date.month)
        day = int(date.day)
        hour = int(date.hour)

        #Take just observations that match month, day, and hour group in the past
        sub_df = weather[(weather['Month']==month) & (weather['Day']==day) & (weather['HourGroup']==hour)]

        #Generate average weather from historical data
        avg_C = np.mean(sub_df['C'])
        avg_ms = np.mean(sub_df['m/s'])
        avg_PRCP = np.mean(sub_df['PRCP'])
        avg_SNOW = np.mean(sub_df['SNOW'])

        #Append 
        future_weather=future_weather.append({'Year':year, 'C':avg_C, 'm/s': avg_ms,
                               'Month':month, 'Day':day, 'HourGroup': hour,
                               'PRCP': avg_PRCP, 'SNOW':avg_SNOW}, ignore_index=True)

        #Increment the date
        date = date + datetime.timedelta(hours = hour_delta)

        #Reset date variables for new date
        year = int(date.year)
        month = int(date.month)
        day = int(date.day)
        hour = int(date.hour)

    

    #Now we fill in the gaps between the hour groups

    #df for extra rows
    extra_rows = pd.DataFrame({})
        
    for i, row in future_weather.iterrows():

        hour_group = row['HourGroup']
        year = row['Year']
        day = row['Day']
        month = row['Month']
        hour = row['HourGroup']

        C = row['C']
        ms = row['m/s']
        PRCP = row['PRCP']
        SNOW = row['SNOW']

        for j in range(1, hour_delta):
            hour = hour_group + j
        
            extra_rows=extra_rows.append({'Year':year, 'C':C, 'm/s': ms,
                                   'Month':month, 'Day':day, 'HourGroup': hour,
                                   'PRCP': PRCP, 'SNOW':SNOW}, ignore_index=True)

    #Combine the two
    future_weather = future_weather.append(extra_rows)

    future_weather.sort_values(by=['Year','Month','Day','HourGroup'], inplace=True)
    future_weather.reset_index(drop=True, inplace=True)
    future_weather.to_csv(f'future_weather_{hour_delta}h.csv')

    return future_weather


########
#Sports#
########

def generate_future_nfl(hour_delta=1):
    '''
    Function to impute future Chicago Bears game times for model input based on historical data

    Parameters
    ----------
    hour_delta (default 1): the difference in hours between each observation

    Returns
    -------
    DataFrame containing imputed game indicator variables until end of 2023
    '''

    #Look for clean historical NFL data
    file_path = f'ChicagoNFL_clean_{hour_delta}h.csv'
    
    if(os.path.isfile(file_path)):
        nfl = pd.read_csv(file_path, index_col=0)
    else:
        nfl = clean_nfl_schedule('ChicagoNFL_raw.xlsx', hour_delta)

    #Also get the new games that we already know are taking place
    file_path = f'future_weather_{hour_delta}h.csv'
    
    if(os.path.isfile(file_path)):
        weather = pd.read_csv(file_path)
    else:
        weather = pd.DataFrame({})

    file_path = f'historical_weather_clean_{hour_delta}h.csv'
    
    if(os.path.isfile(file_path)):
        weather_hist = pd.read_csv(file_path)
    else:
        weather_hist = pd.DataFrame({})

    weather = weather[['Year','Month','Day','HourGroup']]
    weather_hist = weather_hist[['Year','Month','Day','HourGroup']]

    times = weather.append(weather_hist)
    times = times.sort_values(by=['Year','Month','Day','HourGroup'])
    times['BearsGame'] = 0 

    #Finally load the weather data so we can grab timestamps
    file_path = 'FutureNFL.xlsx'
    
    if(os.path.isfile(file_path)):
        nfl_upcoming = pd.read_excel(file_path)
    else:
        nfl_upcoming = pd.DataFrame({})  

    #Match up nfl_upcoming to historical games
    nfl_upcoming = nfl_upcoming[['Date']]

    nfl_upcoming['Date'] = pd.to_datetime(nfl_upcoming['Date'])
    nfl_upcoming['Year'] = nfl_upcoming['Date'].dt.year
    nfl_upcoming['Month'] = nfl_upcoming['Date'].dt.month
    nfl_upcoming['Day'] = nfl_upcoming['Date'].dt.day
    #nfl_upcoming['Weekday'] = nfl_upcoming['Date'].dt.dayofweek
    nfl_upcoming['HourGroupStart'] = nfl_upcoming['Date'].dt.hour

    #Duration of average game is 192 minutes, so will end in HourGroup16
    nfl_upcoming['HourGroupEnd'] = 16
    nfl_upcoming['Team'] = 'BEARS'

    #Do we already have a times?
    if(os.path.isfile('sports_times.csv')):
        times = pd.read_csv('sports_times.csv', index_col=0)
    else:
        #Make a date column for times
        times['Weekday'] = 0

        times.drop_duplicates(inplace=True)
        times.reset_index(drop=True, inplace=True)
        
        for i, row in times.iterrows():
            year = str(int(row['Year']))

            
            month = str(int(row['Month']))
            if(len(month)==1):
                month = "0"+month
                
            day = str(int(row['Day']))
            if(len(day)==1):
                day = "0"+day
                      
            hour = str(int(row['HourGroup']))

            date_str = f'{month}/{day}/{year} {hour}:00:00'
            date = datetime.datetime.strptime(date_str, "%m/%d/%Y %H:%M:%S")

            times.loc[i,'Weekday'] = date.weekday()

        times.to_csv('sports_times.csv')
    

    #Append to historical data
    nfl = nfl.append(nfl_upcoming)
    nfl.drop(columns='Date', inplace=True)
    nfl.reset_index(drop=True, inplace=True)

    #Expand to hourly for NFL
    for i, row in nfl.iterrows():
        year = row['Year']
        month = row['Month']
        day = row['Day']

        hour_start = int(row['HourGroupStart'])
        hour_end = int(row['HourGroupEnd'])


        times.loc[(times['Year']==year) & (times['Month']==month) & (times['Day']==day) & (times['HourGroup']>=hour_start) & (times['HourGroup']<=hour_end), 'BearsGame'] = 1
        

    #Now we split out and impute for dates after 1/8/2023 where scheduled games stop
    times_cutoff_index = times.loc[(times['Year']==2023) & (times['Month']==1) & (times['Day']==8)].index[-1]+1

    times_old = times.iloc[:times_cutoff_index,]
    times_future = times.iloc[times_cutoff_index:,]

    for i, row in times_future.iterrows():

        year = row['Year']
        month = row['Month']
        dow = row['Weekday']
        time = row['HourGroup']

        times_future.loc[i, 'BearsGame'] = np.mean(times_old.loc[(times_old['Month']==month)&(times_old['Weekday']==dow)&(times_old['HourGroup']==time),'BearsGame'])


    #For the 8 (home game count) most likely games that will take place in 2023, we assign a 1
    likely_games = times_future.groupby(by=['Year','Month','Day']).mean().sort_values(by='BearsGame').reset_index().iloc[-8:,]
    times_future['BearsGame']= 0

    #Assume that games typically start at 1PM ET
    for i, row in likely_games.iterrows():
        year = row['Year']
        month = row['Month']
        day = row['Day']
        

        times_future.loc[(times_future['Year']==year)&(times_future['Month']==month)&(times_future['Day']==day)&(times_future['HourGroup']>=12)&(times_future['HourGroup']<=16), 'BearsGame'] = 1
        
    times = times_old.append(times_future)
        
    return times

def generate_future_nba(hour_delta=1):
    '''
    Function to impute future Chicago Bulls game times for model input based on historical data

    Parameters
    ----------
    hour_delta (default 1): the difference in hours between each observation

    Returns
    -------
    DataFrame containing imputed game indicator variables until end of 2023
    '''

    #Look for clean historical NFL data
    file_path = f'ChicagoNBA_clean_{hour_delta}h.csv'
    
    if(os.path.isfile(file_path)):
        nba = pd.read_csv(file_path, index_col=0)
    else:
        nba = clean_nba_schedule('ChicagoNBA_raw.xlsx', hour_delta)

    #Also get the new games that we already know are taking place
    file_path = f'future_weather_{hour_delta}h.csv'
    
    if(os.path.isfile(file_path)):
        weather = pd.read_csv(file_path)
    else:
        weather = pd.DataFrame({})

    file_path = f'historical_weather_clean_{hour_delta}h.csv'
    
    if(os.path.isfile(file_path)):
        weather_hist = pd.read_csv(file_path)
    else:
        weather_hist = pd.DataFrame({})

    weather = weather[['Year','Month','Day','HourGroup']]
    weather_hist = weather_hist[['Year','Month','Day','HourGroup']]

    times = weather.append(weather_hist)
    times = times.sort_values(by=['Year','Month','Day','HourGroup'])
    times['BullsGame'] = 0 

    #Finally load the weather data so we can grab timestamps
    file_path = 'FutureNBA.xlsx'
    
    if(os.path.isfile(file_path)):
        nba_upcoming = pd.read_excel(file_path)
    else:
        nba_upcoming = pd.DataFrame({})  

    #Match up nba_upcoming to historical games
    nba_upcoming = nba_upcoming[['Date']]

    nba_upcoming['Date'] = pd.to_datetime(nba_upcoming['Date'])
    nba_upcoming['Year'] = nba_upcoming['Date'].dt.year
    nba_upcoming['Month'] = nba_upcoming['Date'].dt.month
    nba_upcoming['Day'] = nba_upcoming['Date'].dt.day
    #nba_upcoming['Weekday'] = nba_upcoming['Date'].dt.dayofweek
    nba_upcoming['HourGroupStart'] = nba_upcoming['Date'].dt.hour

    #Duration of average game is 132 minutes, so will end in HourGroup14
    nba_upcoming['HourGroupEnd'] = nba_upcoming['HourGroupStart']+3
    nba_upcoming['Team'] = 'BULLS'

    #Do we already have a times?
    if(os.path.isfile('sports_times.csv')):
        times = pd.read_csv('sports_times.csv', index_col=0)
        times.drop(columns='BearsGame',inplace=True)
    else:
        #Make a date column for times
        times['Weekday'] = 0

        times.drop_duplicates(inplace=True)
        times.reset_index(drop=True, inplace=True)
        
        for i, row in times.iterrows():
            year = str(int(row['Year']))

            
            month = str(int(row['Month']))
            if(len(month)==1):
                month = "0"+month
                
            day = str(int(row['Day']))
            if(len(day)==1):
                day = "0"+day
                      
            hour = str(int(row['HourGroup']))

            date_str = f'{month}/{day}/{year} {hour}:00:00'
            date = datetime.datetime.strptime(date_str, "%m/%d/%Y %H:%M:%S")

            times.loc[i,'Weekday'] = date.weekday()

        times.to_csv('sports_times.csv')
    
    times['BullsGame']=0
    #Append to historical data
    nba = nba.append(nba_upcoming)
    nba.drop(columns='Date', inplace=True)
    nba.reset_index(drop=True, inplace=True)

    #Expand to hourly for nba
    for i, row in nba.iterrows():
        year = row['Year']
        month = row['Month']
        day = row['Day']

        hour_start = int(row['HourGroupStart'])
        hour_end = int(row['HourGroupEnd'])


        times.loc[(times['Year']==year) & (times['Month']==month) & (times['Day']==day) & (times['HourGroup']>=hour_start) & (times['HourGroup']<=hour_end), 'BullsGame'] = 1
        

    #Now we split out and impute for dates after 4/9/2023 where scheduled games stop
    times_cutoff_index = times.loc[(times['Year']==2023) & (times['Month']==4) & (times['Day']==23)].index[-1]+1

    times_old = times.iloc[:times_cutoff_index,]
    times_future = times.iloc[times_cutoff_index:,]

    for i, row in times_future.iterrows():

        year = row['Year']
        month = row['Month']
        dow = row['Weekday']
        time = row['HourGroup']

        times_future.loc[i, 'BullsGame'] = np.mean(times_old.loc[(times_old['Month']==month)&(times_old['Weekday']==dow)&(times_old['HourGroup']==time),'BullsGame'])

    #For the 41 (home) most likely games that will take place in 2023, we assign a 1
    likely_games = times_future.groupby(by=['Year','Month','Day']).mean().sort_values(by='BullsGame').reset_index().iloc[-41:,]
    times_future['BullsGame']= 0

    #Assume that games typically start at 7:30PM ET on average
    for i, row in likely_games.iterrows():
        year = row['Year']
        month = row['Month']
        day = row['Day']
        

        times_future.loc[(times_future['Year']==year)&(times_future['Month']==month)&(times_future['Day']==day)&(times_future['HourGroup']>=18)&(times_future['HourGroup']<=20), 'BullsGame'] = 1
        
    times = times_old.append(times_future)
        
    return times

def generate_future_chc(hour_delta=1):
    '''
    Function to impute future Chicago Cubs game times for model input based on historical data

    Parameters
    ----------
    hour_delta (default 1): the difference in hours between each observation

    Returns
    -------
    DataFrame containing imputed game indicator variables until end of 2023
    '''
    #Look for clean historical mlb data
    file_path = f'ChicagoCHC_clean_{hour_delta}h.csv'
    
    if(os.path.isfile(file_path)):
        mlb = pd.read_csv(file_path, index_col=0)

    #Also get the new games that we already know are taking place
    file_path = f'future_weather_{hour_delta}h.csv'
    
    if(os.path.isfile(file_path)):
        weather = pd.read_csv(file_path)
    else:
        weather = pd.DataFrame({})

    file_path = f'historical_weather_clean_{hour_delta}h.csv'
    
    if(os.path.isfile(file_path)):
        weather_hist = pd.read_csv(file_path)
    else:
        weather_hist = pd.DataFrame({})

    weather = weather[['Year','Month','Day','HourGroup']]
    weather_hist = weather_hist[['Year','Month','Day','HourGroup']]

    times = weather.append(weather_hist)
    times = times.sort_values(by=['Year','Month','Day','HourGroup'])
    times['CubsGame'] = 0 
    
    #Do we already have a times?
    if(os.path.isfile('sports_times.csv')):
        times = pd.read_csv('sports_times.csv', index_col=0)
        times.drop(columns='BearsGame',inplace=True)
    else:
        #Make a date column for times
        times['Weekday'] = 0

        times.drop_duplicates(inplace=True)
        times.reset_index(drop=True, inplace=True)
        
        for i, row in times.iterrows():
            year = str(int(row['Year']))

            
            month = str(int(row['Month']))
            if(len(month)==1):
                month = "0"+month
                
            day = str(int(row['Day']))
            if(len(day)==1):
                day = "0"+day
                      
            hour = str(int(row['HourGroup']))

            date_str = f'{month}/{day}/{year} {hour}:00:00'
            date = datetime.datetime.strptime(date_str, "%m/%d/%Y %H:%M:%S")

            times.loc[i,'Weekday'] = date.weekday()

        times.to_csv('sports_times.csv')
    
    times['CubsGame']=0
    mlb.reset_index(drop=True, inplace=True)

    #Expand to hourly for mlb
    for i, row in mlb.iterrows():
        year = row['Year']
        month = row['Month']
        day = row['Day']

        hour_start = int(row['HourGroupStart'])
        hour_end = int(row['HourGroupEnd'])


        times.loc[(times['Year']==year) & (times['Month']==month) & (times['Day']==day) & (times['HourGroup']>=hour_start) & (times['HourGroup']<=hour_end), 'CubsGame'] = 1
        
    #Now we split out and impute for dates after 10/5/2022 where scheduled games stop
    times_cutoff_index = times.loc[(times['Year']==2022) & (times['Month']==10) & (times['Day']==5)].index[-1]+1

    times_old = times.iloc[:times_cutoff_index,]
    times_future = times.iloc[times_cutoff_index:,]

    for i, row in times_future.iterrows():

        year = row['Year']
        month = row['Month']
        dow = row['Weekday']
        time = row['HourGroup']

        times_future.loc[i, 'CubsGame'] = np.mean(times_old.loc[(times_old['Month']==month)&(times_old['Weekday']==dow)&(times_old['HourGroup']==time),'CubsGame'])

    #For the 81 (home) most likely games that will take place in 2023, we assign a 1
    likely_games = times_future.groupby(by=['Year','Month','Day']).mean().sort_values(by='CubsGame').reset_index().iloc[-81:,]
    times_future['CubsGame']= 0

    #Assume that games typically start at 1:20PM ET on average for the Cubs
    for i, row in likely_games.iterrows():
        year = row['Year']
        month = row['Month']
        day = row['Day']
        
        
        times_future.loc[(times_future['Year']==year)&(times_future['Month']==month)&(times_future['Day']==day)&(times_future['HourGroup']>=13)&(times_future['HourGroup']<=16), 'CubsGame'] = 1
        
    times = times_old.append(times_future)
    
    return times

def generate_future_cws(hour_delta=1):
    '''
    Function to impute future Chicago White Sox game times for model input based on historical data

    Parameters
    ----------
    hour_delta (default 1): the difference in hours between each observation

    Returns
    -------
    DataFrame containing imputed game indicator variables until end of 2023
    '''

    #Look for clean historical mlb data
    file_path = f'ChicagoCHW_clean_{hour_delta}h.csv'
    
    if(os.path.isfile(file_path)):
        mlb = pd.read_csv(file_path, index_col=0)

    #Also get the new games that we already know are taking place
    file_path = f'future_weather_{hour_delta}h.csv'
    
    if(os.path.isfile(file_path)):
        weather = pd.read_csv(file_path)
    else:
        weather = pd.DataFrame({})

    file_path = f'historical_weather_clean_{hour_delta}h.csv'
    
    if(os.path.isfile(file_path)):
        weather_hist = pd.read_csv(file_path)
    else:
        weather_hist = pd.DataFrame({})

    weather = weather[['Year','Month','Day','HourGroup']]
    weather_hist = weather_hist[['Year','Month','Day','HourGroup']]

    times = weather.append(weather_hist)
    times = times.sort_values(by=['Year','Month','Day','HourGroup'])
    times['SoxGame'] = 0 
    
    #Do we already have a times?
    if(os.path.isfile('sports_times.csv')):
        times = pd.read_csv('sports_times.csv', index_col=0)
        times.drop(columns='BearsGame',inplace=True)
    else:
        #Make a date column for times
        times['Weekday'] = 0

        times.drop_duplicates(inplace=True)
        times.reset_index(drop=True, inplace=True)
        
        for i, row in times.iterrows():
            year = str(int(row['Year']))

            
            month = str(int(row['Month']))
            if(len(month)==1):
                month = "0"+month
                
            day = str(int(row['Day']))
            if(len(day)==1):
                day = "0"+day
                      
            hour = str(int(row['HourGroup']))

            date_str = f'{month}/{day}/{year} {hour}:00:00'
            date = datetime.datetime.strptime(date_str, "%m/%d/%Y %H:%M:%S")

            times.loc[i,'Weekday'] = date.weekday()

        times.to_csv('sports_times.csv')
    
    times['SoxGame']=0
    mlb.reset_index(drop=True, inplace=True)

    #Expand to hourly for mlb
    for i, row in mlb.iterrows():
        year = row['Year']
        month = row['Month']
        day = row['Day']

        hour_start = int(row['HourGroupStart'])
        hour_end = int(row['HourGroupEnd'])


        times.loc[(times['Year']==year) & (times['Month']==month) & (times['Day']==day) & (times['HourGroup']>=hour_start) & (times['HourGroup']<=hour_end), 'SoxGame'] = 1
        
    #Now we split out and impute for dates after 10/5/2022 where scheduled games stop
    times_cutoff_index = times.loc[(times['Year']==2022) & (times['Month']==10) & (times['Day']==5)].index[-1]+1

    times_old = times.iloc[:times_cutoff_index,]
    times_future = times.iloc[times_cutoff_index:,]

    for i, row in times_future.iterrows():

        year = row['Year']
        month = row['Month']
        dow = row['Weekday']
        time = row['HourGroup']

        times_future.loc[i, 'SoxGame'] = np.mean(times_old.loc[(times_old['Month']==month)&(times_old['Weekday']==dow)&(times_old['HourGroup']==time),'SoxGame'])

    #For the 81 (home) most likely games that will take place in 2023, we assign a 1
    likely_games = times_future.groupby(by=['Year','Month','Day']).mean().sort_values(by='SoxGame').reset_index().iloc[-81:,]
    times_future['SoxGame']= 0

    #Assume that games typically start at 1:10PM ET on average for the Cubs
    for i, row in likely_games.iterrows():
        year = row['Year']
        month = row['Month']
        day = row['Day']
        
        
        times_future.loc[(times_future['Year']==year)&(times_future['Month']==month)&(times_future['Day']==day)&(times_future['HourGroup']>=13)&(times_future['HourGroup']<=16), 'SoxGame'] = 1
        
    times = times_old.append(times_future)
    
    return times


#Get sports starting 11/7 (end of historical data)
a = generate_future_nba().iloc[181349:,].reset_index(drop=True)
b = generate_future_nfl().iloc[181349:,].reset_index(drop=True)
c = generate_future_chc().iloc[181349:,].reset_index(drop=True)
d = generate_future_cws().iloc[181349:,].reset_index(drop=True)

#Generate future weather
e = generate_future_weather()

#Join altogether and save to csv
a.join([b['BearsGame'],e[['C','m/s','PRCP','SNOW']],c['CubsGame'],d['SoxGame']]).to_csv('future_data_full.csv')
