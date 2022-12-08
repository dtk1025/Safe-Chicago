#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 15:46:22 2022

@author: deez
"""

# make sure to install these packages before running:
# pip install pandas
# pip install sodapy

import pandas as pd
from sodapy import Socrata

def get_latest_date(input_df):
    print ('Start to convert Date column to datetime')
    input_df["Date"] = pd.to_datetime(input_df["Date"])
    max_date = max(input_df['Date'])
    max_date_str = max_date.strftime("%Y-%m-%dT%H:%M:%S.000")
    max_date_str = "\"" + max_date_str + "\""
    return max_date_str

# Ensure crimes_raw.csv (can get it from the team google drive) is in the folder 
# Call a function to get the latest date


print('Start to read crimes_raw dataset')
df = pd.read_csv('crimes_raw.csv')
print('Total rows: ', len(df))

latest_date = get_latest_date(df)
print("latest date: ", latest_date)


# Since we already know the latest date is 2022-10-06T23:42:00.000 for the crimes_raw.csv, just set the date
latest_date = "\"2022-10-06T23:42:00.000\""

client = Socrata("data.cityofchicago.org", "iClKK3Jg0HlPv7IUnWP0XOVEn",timeout=1000)


results = client.get("ijzp-q8t2", where="date > "+ latest_date , limit=1000000 )

# Convert to pandas DataFrame
api_return_df = pd.DataFrame.from_records(results)

# output to csv
print('Total crimes return from API: ', len(api_return_df))
api_return_df.to_csv('api_return_df.csv')

#Append new rows to original crimes_raw
api_return_df.append(df, ignore_index=True, inplace=True)


# looking at all columns and total null count
pd.set_option('display.max_columns', None)
api_return_df[api_return_df.isnull().any(axis=1)].count()

# drop nans and check nan count
api_return_df.dropna(inplace=True)
print("N/A Counts per column")
api_return_df.isnull().sum(axis=0)

# change df name to crimes for consistentcy
crimes = api_return_df

# drop unnecessary columns
drop_cols = ['case_number','iucr','ward','fbi_code',
             'updated_on','location', 'x_coordinate', 
             'y_coordinate']

crimes.drop(drop_cols, axis=1, inplace=True)

# align column names to the notebook. Block is the only additional column added
crimes.set_axis(['ID', 'Date', 'Block', 'Primary Type', 'Description', 'Location Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Community Area', 'Year', 'Latitude', 'Longitude'], axis=1, inplace=True)


# convert to datetime to match datetime format in notebook
crimes['Date']=pd.to_datetime(crimes['Date'])
crimes['Date_Fix'] = pd.to_datetime(crimes['Date'])
crimes['Month'] = crimes['Date'].dt.month
crimes['Time'] = crimes['Date'].dt.time
crimes['Day'] = crimes['Date'].dt.date
crimes['Week'] = crimes['Date']-pd.to_timedelta(7, unit='d')






