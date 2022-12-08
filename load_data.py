'''
Functions for loading and cleaning data
'''

#Importing libraries
import os
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import numpy as np
import math
from sodapy import Socrata
import datetime
import dateutil.parser
import requests
import re
from weather_format import *
from itertools import product
import matplotlib.pyplot as plot
import time
import holidays
import geocoder
import json
import geojson
from shapely.geometry import shape, Point

def get_max_date_socrata(df):
    """Find the max of the Date column of a DataFrame as a string
    Parameters
    ----------
    df: DataFrame with Date column
    
    Returns
    -------
    Max of the Date column of a DataFrame as a string
    """
    return max(pd.to_datetime(df["Date"])).strftime('"%Y-%m-%dT%H:%M:%S.000"')

def get_current_crimes_data(from_date):
    """Access new Chicago Crimes data
    Parameters
    ----------
    from_date: get data that is more current than from_date
    
    Returns
    -------
    DataFrame of new Chicago Crimes data since from_date
    """
    client = Socrata(domain="data.cityofchicago.org", app_token="iClKK3Jg0HlPv7IUnWP0XOVEn", timeout=100)
    results = client.get(dataset_identifier="ijzp-q8t2", where="date > "+ from_date, limit=1000000)
    return pd.DataFrame.from_records(results)

def clean_nba_schedule(file_path, hour_delta):
  """Clean up NBA schedule for the Bulls
  Parameters
  ----------
  file_path: file path to raw schedule Excel file
  
  Returns
  -------
  DataFrame with date, start time, duration
  """
  #Read in raw data
  schedule_df = pd.read_excel(file_path)

  #Drop unnecessary columns
  schedule_df = schedule_df[['Date','Start (ET)']]

  #Format the start time field
  schedule_df['Start (ET)'] = schedule_df.apply(lambda row: row['Start (ET)'][:-1], axis=1)
  schedule_df['Start (ET)'] = schedule_df.apply(lambda row: datetime.datetime.strptime(row['Start (ET)'], "%H:%M"), axis=1)
  schedule_df['Start (ET)'] = schedule_df.apply(lambda row: row['Start (ET)'].time(), axis=1)

  #Format the date
  schedule_df['Date'] = schedule_df.apply(lambda row: row['Date'][5:], axis=1)
  schedule_df['Date'] = schedule_df.apply(lambda row: datetime.datetime.strptime(row['Date'], "%b %d, %Y"),axis=1)

  #Combine into a datetime field
  schedule_df['StartTime'] = schedule_df.apply(lambda row: datetime.datetime.combine(row['Date'],row['Start (ET)']).time(), axis=1)

  #Median length of NBA games are 2 hours and 12 minutes, set since we don't have duration of games
  schedule_df = schedule_df.assign(Duration=132)
  
  #Keep only these columns
  schedule_df = schedule_df[['Date','StartTime','Duration']]
  schedule_df = schedule_df.assign(Team='BULLS')

  #Reform to match formatting of aggregated crimes date
  schedule_df['StartTime'] = schedule_df['Date'].astype(str) + " " + schedule_df['StartTime'].astype(str) + "PM"
  schedule_df['StartTime'] = schedule_df['StartTime'].apply(lambda val: dateutil.parser.parse(val))
  schedule_df['EndTime'] = schedule_df.apply(lambda row: row['StartTime'] + pd.Timedelta(minutes=row['Duration']),axis=1)

  #Change over to hour group
  schedule_df['HourGroupStart'] = (schedule_df['StartTime'].dt.hour//hour_delta) * hour_delta
  schedule_df['HourGroupEnd'] = (schedule_df['EndTime'].dt.hour//hour_delta) * hour_delta

  #Break out Date
  schedule_df['Year'] = schedule_df['Date'].dt.year
  schedule_df['Month'] = schedule_df['Date'].dt.month
  schedule_df['Day'] = schedule_df['Date'].dt.day

  #Get rid of columns
  schedule_df = schedule_df[['Year','Month','Day','HourGroupStart','HourGroupEnd','Team']]

  #Save to csv
  schedule_df.to_csv(f'ChicagoNBA_clean_{hour_delta}h.csv')

  return schedule_df

def get_mlb_schedule(team_label, hour_delta):
  """Retrieve up MLB schedule for a given team
  Parameters
  ----------
  team_label: team label e.g., 'CHC', 'CHW'
  
  Returns
  -------
  DataFrame with date, start time, duration
  """
  schedule_df = pd.DataFrame()

  #What's the date?
  today = datetime.date.today()
  year = today.year
  date = today.day
  month = today.month

  #Query baseballreference.com for the stats on the specified team for each year
  for i in range(2001, year+1):
    pull = pd.read_html(f'https://www.baseball-reference.com/teams/{team_label}/{i}-schedule-scores.shtml#team_schedule')[0]
    pull['Year'] = i
    schedule_df = pd.concat([schedule_df, pull], axis=0)
    time.sleep(2.0)

  #This gets rid of headers from the HTML table
  schedule_df = schedule_df[schedule_df['Attendance'] != 'Attendance']
  schedule_df = schedule_df[schedule_df['Date'] != 'Date']

  #This keeps only home games
  schedule_df = schedule_df[schedule_df['Unnamed: 4'] != '@']

  #This keeps just the columns that we are in need of 
  schedule_df = schedule_df[['Date','Time','Attendance','Year','D/N']]

  #Transform date and year into datetime object
  schedule_df['Date'] = schedule_df.apply(lambda row: row['Date'].split("(")[0], axis=1)
  schedule_df['FullDate'] = schedule_df['Date'] + ' ' + schedule_df['Year'].astype(str)
  schedule_df['Date'] = pd.to_datetime(schedule_df['FullDate'])

  #Drop the year column
  schedule_df.drop(['FullDate'],axis=1, inplace=True)

  #Recode the D/N 
  schedule_df['D/N'] = schedule_df.apply(lambda row: 1 if (row["D/N"] == 'N') else 0, axis=1)

  #Set the StartTime
  if(team_label=='CHC'):
    schedule_df['StartTime'] = schedule_df.apply(lambda row: datetime.datetime.strptime("1:20 PM","%I:%M %p").time() if row['D/N']==0 else datetime.datetime.strptime("6:45 PM","%I:%M %p").time(), axis=1)
  else:
    schedule_df['StartTime'] = schedule_df.apply(lambda row: datetime.datetime.strptime("1:10 PM","%I:%M %p").time() if row['D/N']==0 else datetime.datetime.strptime("6:45 PM","%I:%M %p").time(), axis=1)
    
  schedule_df['Team'] = team_label

  #Change the duration into minutes
  schedule_df['Duration'] = schedule_df.apply(lambda row: 60*int(row['Time'].split(":")[0]) + int(row['Time'].split(":")[1]), axis=1) 

  #Drop the Total time, attendance, year, d/n
  schedule_df.drop(columns=['Time','Attendance','Year','D/N'], inplace=True)

  #Break out Date
  schedule_df['Year'] = schedule_df['Date'].dt.year
  schedule_df['Month'] = schedule_df['Date'].dt.month
  schedule_df['Day'] = schedule_df['Date'].dt.day

  #Reform to match formatting of aggregated crimes date
  schedule_df['StartTime'] = schedule_df['Date'].astype(str) + " " + schedule_df['StartTime'].astype(str)
  schedule_df['StartTime'] = schedule_df['StartTime'].apply(lambda val: dateutil.parser.parse(val))
  schedule_df['EndTime'] = schedule_df.apply(lambda row: row['StartTime'] + pd.Timedelta(minutes=row['Duration']),axis=1)

  #Change over to hour group
  schedule_df['HourGroupStart'] = (schedule_df['StartTime'].dt.hour//hour_delta) * hour_delta
  schedule_df['HourGroupEnd'] = (schedule_df['EndTime'].dt.hour//hour_delta) * hour_delta

  #Get rid of columns that aren't necessary
  schedule_df = schedule_df[['Year','Month','Day','HourGroupStart','HourGroupEnd','Team']]

  schedule_df.to_csv(f'Chicago{team_label}_clean_{hour_delta}h.csv')

  return schedule_df

def clean_nfl_schedule(file_path, hour_delta):
  """Clean up NFL schedule for the Bears
  Parameters
  ----------
  file_path: file path to raw schedule Excel file
  
  Returns
  -------
  DataFrame with date, start time, duration
  """
  #Read in raw data
  schedule_df = pd.read_excel(file_path)

  #Drop unnecessary columns
  schedule_df = schedule_df[['Date','Start Time']]

  #Format the start time field
  schedule_df['Start Time'] = schedule_df.apply(lambda row: row['Start Time'][:-3], axis=1)
  schedule_df['Start Time'] = schedule_df.apply(lambda row: datetime.datetime.strptime(row['Start Time'], "%I:%M%p"), axis=1)
  schedule_df['Start Time'] = schedule_df.apply(lambda row: row['Start Time'].time(), axis=1)

  #Format the date
  schedule_df['Date'] = schedule_df['Date'].map(str)
  schedule_df['Date'] = schedule_df.apply(lambda row: datetime.datetime.strptime(row['Date'], "%Y-%m-%d %H:%M:%S"),axis=1)

  #Combine into a datetime field
  schedule_df['StartTime'] = schedule_df.apply(lambda row: datetime.datetime.combine(row['Date'],row['Start Time']).time(), axis=1)
  #schedule_df['StartTime'] = schedule_df['Start Time'].time()
  
  #Median length of NFL games are 3 hours and 12 minutes, set since we don't have duration of games
  schedule_df = schedule_df.assign(Duration=192)

  #Keep only these columns
  schedule_df = schedule_df[['Date','StartTime','Duration']]
  schedule_df = schedule_df.assign(Team='BEARS')

  #Reform to match formatting of aggregated crimes date
  schedule_df['StartTime'] = schedule_df['Date'].astype(str) + " " + schedule_df['StartTime'].astype(str)
  schedule_df['StartTime'] = schedule_df['StartTime'].apply(lambda val: dateutil.parser.parse(val))
  schedule_df['EndTime'] = schedule_df.apply(lambda row: row['StartTime'] + pd.Timedelta(minutes=row['Duration']),axis=1)

  #Change over to hour group
  schedule_df['HourGroupStart'] = (schedule_df['StartTime'].dt.hour//hour_delta) * hour_delta
  schedule_df['HourGroupEnd'] = (schedule_df['EndTime'].dt.hour//hour_delta) * hour_delta

  #Break out Date
  schedule_df['Year'] = schedule_df['Date'].dt.year
  schedule_df['Month'] = schedule_df['Date'].dt.month
  schedule_df['Day'] = schedule_df['Date'].dt.day

  #Get rid of columns
  schedule_df = schedule_df[['Year','Month','Day','HourGroupStart','HourGroupEnd','Team']]

  schedule_df.to_csv(f'ChicagoNFL_clean_{hour_delta}h.csv')
  return schedule_df


def clean_crimes(file_path):
    """
    Clean up crimes_raw data set
    Parameters
    ----------
    file_path: file path to full crimes_raw csv file
      
    Returns
    -------
    DataFrame colums for crime categorization, logcation, and time of occurence
    """
    crimes_raw = pd.read_csv(file_path)

    #We know the most up-to-date is "\"2022-10-06T23:42:00.000\"" in raw file
    current_crimes_raw = get_current_crimes_data("\"2022-10-06T23:42:00.000\"")

    current_crimes_raw.columns = crimes_raw.columns
    crimes = pd.concat([crimes_raw, current_crimes_raw],axis=0, ignore_index=True)
    
    #Drop useless columns
    drop_cols = ['Case Number','IUCR','Ward','FBI Code',
             'Updated On','Location', 'X Coordinate', 
             'Y Coordinate', 'Location Description']
    crimes.drop(drop_cols, axis=1, inplace=True)

    #Format dates
    crimes['Date']=pd.to_datetime(crimes['Date'])
    crimes['Year'] = crimes['Date'].dt.year
    crimes['Month'] = crimes['Date'].dt.month
    crimes['Week'] = crimes['Date']-pd.to_timedelta(7, unit='d')
    crimes['Day'] = crimes['Date'].dt.day
    crimes['Time'] = crimes['Date'].dt.time
    crimes['Hour'] = crimes['Date'].dt.hour

    #Drop before 2003 due to very high # of missing values that take an
    #exceedingly long time to refill via function below
    crimes = crimes[crimes['Year'] >= 2003]
    
    #Replace any missing community area fields
    with open('Boundaries - Community Areas (current).geojson') as f:
        boundaries = json.load(f)
    
    crimes_na = crimes[crimes['Community Area'].isna()]
    crimes_na = crimes_na[crimes_na['Latitude'].notna()]
    #crimes_na['Community Area'] = crimes_na.apply(lambda row: (geocode_geojson(row['Latitude'], row['Longitude'], boundaries)), axis=1)

    for i, row in crimes_na.iterrows():
        crimes.loc[i, 'Community Area'] = geocode_geojson(crimes_na.loc[i, 'Latitude'], crimes_na.loc[i, 'Longitude'], boundaries)


    #Drop columns with important missing data that we can't impute after wards
    crimes = crimes[crimes['Community Area'].notna()]

    crimes.sort_values(by='Date', inplace=True)
    crimes.reset_index(inplace=True)
           
    #Save out clean file
    crimes.to_csv('crimes_clean.csv')

    return crimes


def aggregate_crimes(file_path, hour_delta):
    """
    Aggregates crime counts within the crimes dataset into specified time periods
    Parameters
    ----------
    file_path: file path to full crimes_clean csv file
    hour_delta: number of hours to group together from time zero per row
                must be whole number integer cleanly divisible into 24
    Returns
    -------
    DataFrame colums for crime categorization, logcation, grouped by time of occurence
    """

    #reading in crimes dataset
    crimes = pd.read_csv(file_path)

    #Group by
    crimes['HourGroup'] = ((crimes['Hour'] // hour_delta) * hour_delta).astype(int)

    crimes_agg = crimes.groupby(['Community Area','Year','Month','Day','HourGroup'])
    crimes_agg_count = crimes_agg.count()['ID']
    crimes_agg_min = crimes_agg.mean()

    #Reformat columns and combine counts with relevant dates
    crimes_agg = pd.concat([crimes_agg_count,crimes_agg_min], axis=1).reset_index()
    crimes_agg = crimes_agg[['Community Area','HourGroup','ID','Year','Month','Day',]]
    crimes_agg.columns = ['Community Area','HourGroup','Count','ID','Year','Month','Day',]
    crimes_agg.drop(crimes_agg.columns[3], axis=1, inplace=True)

    #Re-implement datetime column
    crimes_agg['DateTime'] = crimes_agg['Year'].astype(str) + "-" + crimes_agg['Month'].astype(str) + "-" +crimes_agg['Day'].astype(str) + " " + crimes_agg['HourGroup'].astype(str) + ":00:00"
    crimes_agg['DateTime'] = pd.to_datetime(crimes_agg['DateTime'])

    
    #Filter out 2001 due to underreporting
    crimes_agg = crimes_agg[crimes_agg['Year'] > 2001]

    #Save to csv
    crimes_agg.to_csv(f'crimes_agg_{hour_delta}h.csv')
    
    #Create all the date permutations that can be possible given the hour delta
    start = datetime.date(2001, 1, 1)

    #Set most recent date in aggregated file
    end = max(pd.to_datetime(crimes_agg['DateTime'])).date()
    date_diff = end-start
    date_list = []

    #Create a dataframe of all possible times starting from the first date available
    for i in range(date_diff.days + 1):
        date_list.append(start + datetime.timedelta(days=i))

    crimes_times = pd.DataFrame(product(date_list,
                                        crimes_agg['HourGroup'].unique()))

    crimes_times.columns = ['Date','HourGroup']
 
    crimes_times['DateTime'] = crimes_times['Date'].astype(str) + " " + crimes_times['HourGroup'].astype(str) + ":00:00"
    crimes_times['DateTime'] = pd.to_datetime(crimes_times['DateTime'])

    crimes_times['Year'] = crimes_times['DateTime'].dt.year.astype(int)
    crimes_times['Month'] = crimes_times['DateTime'].dt.month.astype(int)
    crimes_times['Day'] = crimes_times['DateTime'].dt.day.astype(int)

    #Save to csv
    crimes_times.to_csv(f'crimes_times_{hour_delta}h.csv')

    return crimes_agg

def get_beat_data(agg_hours, beat_num):
    """
    Gets the full data for a single police beat within Chicago
    Parameters
    ----------
    agg_hours: # of hours aggregated together for each observation
    beat_num: the beat data is being pulled for
    
    Returns
    -------
    DataFrame colums for crime categorization, logcation, grouped by time of occurence
    """

    file_path = f'crimes_agg_{agg_hours}h.csv'
    crimes = pd.DataFrame()
    
    #Select the crimes taking place in that beat
    if(os.path.isfile(file_path)):
        crimes = pd.read_csv(file_path)
    else:
        crimes = aggregate_crimes('crimes_clean.csv', int(file_path.split("_")[2].split("h")[0]))

    crimes = crimes[crimes['Community Area'] == beat_num]

    #Get all possible combinations of year/month/day/time
    time_delta_h = file_path.split("_")[2].split(".")[0]
    times = pd.read_csv(f'crimes_times_{time_delta_h}.csv')

    #Get the hour delta
    hour_delta = int(file_path.split("_")[2].split(".")[0][:-1])
    
    #Merge with sports data, generating cleaned data where necessary
    if(os.path.isfile(f'ChicagoNFL_clean_{hour_delta}h.csv')):
        nfl_schedule = pd.read_csv(f'ChicagoNFL_clean_{hour_delta}h.csv')
    else:
        nfl_schedule = clean_nfl_schedule('ChicagoNFL_raw.xlsx', hour_delta)

    if(os.path.isfile('ChicagoNBA_clean_{hour_delta}h.csv')):
        nba_schedule = pd.read_csv(f'ChicagoNBA_clean_{hour_delta}h.csv')
    else:
        nba_schedule = clean_nba_schedule('ChicagoNBA_raw.xlsx', hour_delta)

    if(os.path.isfile(f'ChicagoCHC_clean_1h.csv') & os.path.isfile(f'ChicagoCHW_clean_1h.csv')):
        mlb_schedule = pd.concat([pd.read_csv(f'ChicagoCHC_clean_1h.csv'), pd.read_csv(f'ChicagoCHW_clean_1h.csv')])
    else:
        mlb_schedule = pd.concat([get_mlb_schedule('CHC', 4), get_mlb_schedule('CHW', 4)], axis=0)

    sports_schedule = pd.concat([nba_schedule, nfl_schedule,mlb_schedule])

    #Default set to 0, 1 will be set below 
    times['BullsGame'] = 0
    times['CubsGame'] = 0
    times['SoxGame'] = 0
    times['BearsGame'] = 0

    #Iterate over all possbile times and determine if a sports game occurs during that period
    for i, row in times.iterrows():
        overlap_df = sports_schedule[(sports_schedule.Year == row['Year']) & (sports_schedule.Month == row['Month']) & (sports_schedule.Day == row['Day'])]
        num_overlap = overlap_df.shape[0]

        if(num_overlap > 0):
            #Get the bounds of each game
            temp_game = overlap_df.reset_index()
            for j, sub_row in temp_game.iterrows():
                start = sub_row['HourGroupStart']
                end = sub_row['HourGroupEnd']

                #Set indicator for crimes
                time_hour = row['HourGroup']

                #Set indicator to 1 if game occurs in range
                if((time_hour >= start) & (time_hour <=end)):
                    if(sub_row['Team'] == 'BULLS'):
                        times.loc[i,'BullsGame'] = 1
                    if(sub_row['Team'] == 'BEARS'):
                        times.loc[i,'BearsGame'] = 1
                    if(sub_row['Team'] == 'CHC'):
                        times.loc[i,'CubsGame'] = 1
                    if(sub_row['Team'] == 'CHW'):
                        times.loc[i,'SoxGame'] = 1

    #Merge with weather data; uses different scale by factor of 100
    weather_delta = 100*hour_delta

    #Get weather data or generate it
    if(os.path.isfile(f'historical_weather_clean_{hour_delta}h.csv')):
       weather_data = pd.read_csv(f'historical_weather_clean_{hour_delta}h.csv')
    else:
       weather_data = compile_weather('historical_weather_raw.csv', weather_delta)
    
    #Merge on crimes dataset for time
    times_keys = ['Year','Month','Day','HourGroup']
    weather_keys = ['Year','Month','Day','HourGroup']
    times = times.merge(weather_data, how="left", left_on=times_keys, right_on=times_keys)
    times.drop(columns=['RelHum'],inplace=True)

    #Set DateTime as a dt object
    times['DateTime'] = pd.to_datetime(times['DateTime'])
    crimes['DateTime'] = pd.to_datetime(crimes['DateTime'])
    
    #Merge in crimes counts
    crimes = pd.merge(times, crimes[['DateTime','Count']], how="left", left_on='DateTime', right_on='DateTime')

    #Fill in missing values for counts with zeros (none recorded)
    crimes['Community Area']=beat_num
    crimes = crimes[crimes.columns.drop(list(crimes.filter(regex='Unnamed')))]
    crimes['Count'].fillna(0, inplace=True)

    #Include columns for day of week
    crimes['Weekday'] = pd.to_datetime(crimes['DateTime']).dt.dayofweek

    #Add boolean for federal holiday
    crimes['Date'] = pd.to_datetime(crimes['DateTime']).dt.date
    crimes['Day'] = pd.to_datetime(crimes['DateTime']).dt.day
    crimes['Holiday'] = crimes['Date'].isin(calendar().holidays(start=datetime.date(2001, 1, 1), end=datetime.date(2022, 12, 31)).date)   
    crimes['Holiday'] = crimes.apply(lambda row: 1 if row['Holiday'] == True else 0, axis=1)
    
    #Account for holidays that take place on the weekend not caught by federal holidays
    for i, row in crimes.iterrows():
        holidays = [[12, 25], [12, 31], [7, 4], [1,1], [10, 31], [3,17]]
        for pair in holidays:
            if(row['Month'] == pair[0] and row['Day'] == pair[1]):
                crimes.loc[i, 'Holiday'] = 1

    #Get rid of potential duplicates
    crimes.drop_duplicates(subset='DateTime', inplace=True)
    crimes.sort_values(by='DateTime', inplace=True)
    crimes.reset_index(drop=True, inplace=True)

    
    
    #Imputation of NA values for temp/precip/wind, using monthly average for each historically
    crimes_na_drop = crimes.dropna()

    for i, row in crimes.iterrows():
        if np.isnan(row['C']):
            crimes.loc[i, 'C'] = np.mean(crimes_na_drop[crimes_na_drop.Month == row['Month']]['C'])
        if np.isnan(row['m/s']):
            crimes.loc[i, 'm/s'] = np.mean(crimes_na_drop[crimes_na_drop.Month == row['Month']]['m/s'])
        if np.isnan(row['PRCP']):
            crimes.loc[i, 'PRCP'] = np.mean(crimes_na_drop[crimes_na_drop.Month == row['Month']]['PRCP'])
        if np.isnan(row['SNOW']):
            crimes.loc[i, 'SNOW'] = np.mean(crimes_na_drop[crimes_na_drop.Month == row['Month']]['SNOW'])

    #Drop data before 2003 due to high number of NA values 
    crimes = crimes[crimes['Year'] >= 2003]

    #Save to csv
    crimes.to_csv(f'crimes_agg_ext_beat{beat_num}_{hour_delta}h.csv')
    
    return crimes

def merge_extra_features(crimes_filepath):
    '''
    Function to combine raw crimes data with exogenous features to classifier input data

    '''
    crimes = pd.read_csv(crimes_filepath)
    times = pd.DataFrame({})

    #We won't go through the work of creating the extra features csv if we don't have to
    if not os.path.isfile('sports_weather_hourly.csv'):
        
        #If no 1 hour times file exists, we create one
        if not os.path.isfile(f'crimes_times_1h.csv'):
           aggregate_crimes(crimes_filepath, 1)
        
        times = pd.read_csv('crimes_times_1h.csv')
        
        #Merge with weather data
        weather_delta = 100
        hour_delta = 1

        weather_data = pd.DataFrame({})

        if(os.path.isfile(f'historical_weather_clean_{hour_delta}h.csv')):
           weather_data = pd.read_csv(f'historical_weather_clean_{hour_delta}h.csv')
        else:
           weather_data = compile_weather('historical_weather_raw.csv', weather_delta)

        #We just drop relatively humidity given sparseness
        weather_data.drop(columns='RelHum', inplace=True)

        times_keys = ['Year','Month','Day','HourGroup']
        weather_keys = ['Year','Month','Day','HourGroup']

        #Merge wether data into each possible point in time
        times = times.merge(weather_data, how="left", left_on=times_keys, right_on=weather_keys)

        #We then will merge the sports data onto the weather data for efficiency
        if(os.path.isfile(f'ChicagoNFL_clean_{hour_delta}h.csv')):
            nfl_schedule = pd.read_csv(f'ChicagoNFL_clean_{hour_delta}h.csv')
        else:
            nfl_schedule = clean_nfl_schedule('ChicagoNFL_raw.xlsx', hour_delta)

        if(os.path.isfile('ChicagoNBA_clean_{hour_delta}h.csv')):
            nba_schedule = pd.read_csv(f'ChicagoNBA_clean_{hour_delta}h.csv')
        else:
            nba_schedule = clean_nba_schedule('ChicagoNBA_raw.xlsx', hour_delta)

        if(os.path.isfile(f'ChicagoCHC_clean_{hour_delta}h.csv') & os.path.isfile(f'ChicagoCHW_clean_{hour_delta}h.csv')):
            mlb_schedule = pd.concat([pd.read_csv(f'ChicagoCHC_clean_{hour_delta}h.csv'), pd.read_csv(f'ChicagoCHW_clean_{hour_delta}h.csv')])
        else:
            mlb_schedule = pd.concat([get_mlb_schedule('CHC', 4), get_mlb_schedule('CHW', 4)], axis=0)

        sports_schedule = pd.concat([nba_schedule, nfl_schedule,mlb_schedule])
        
        times['BullsGame'] = 0
        times['CubsGame'] = 0
        times['SoxGame'] = 0
        times['BearsGame'] = 0

        for i, row in times.iterrows():
            overlap_df = sports_schedule[(sports_schedule.Year == row['Year']) & (sports_schedule.Month == row['Month']) & (sports_schedule.Day == row['Day'])]
            num_overlap = overlap_df.shape[0]

            if(num_overlap > 0):
                #Get the bounds of each game
                temp_game = overlap_df.reset_index()
                for j, sub_row in temp_game.iterrows():
                    start = sub_row['HourGroupStart']
                    end = sub_row['HourGroupEnd']

                    #Set indicator for crimes
                    time_hour = row['HourGroup']

                    if((time_hour >= start) & (time_hour <=end)):
                        if(sub_row['Team'] == 'BULLS'):
                            times.loc[i,'BullsGame'] = 1
                        if(sub_row['Team'] == 'BEARS'):
                            times.loc[i,'BearsGame'] = 1
                        if(sub_row['Team'] == 'CHC'):
                            times.loc[i,'CubsGame'] = 1
                        if(sub_row['Team'] == 'CHW'):
                            times.loc[i,'SoxGame'] = 1

        #Imputation of NA values for temp/precip/wind, using monthly average
        times_na_drop = times.dropna()
        times_has_na = times[times['C'].isnull() | times['m/s'].isnull() | times['PRCP'].isnull() | times['SNOW'].isnull()]

        #pre-calculate means for each month
        weather_means = times_na_drop.groupby(by='Month').mean()

        for i, row in times_has_na.iterrows():
            month = row['Month']
            if np.isnan(row['C']):
                times.loc[i, 'C'] = weather_means.loc[month,'C']
            if np.isnan(row['m/s']):
                times.loc[i, 'm/s'] = weather_means.loc[month,'m/s']
            if np.isnan(row['PRCP']):
                times.loc[i, 'PRCP'] = weather_means.loc[month,'PRCP']
            if np.isnan(row['SNOW']):
                times.loc[i, 'SNOW'] = weather_means.loc[month,'SNOW']
        
            #Include columns for weekend
        times['Weekday'] = pd.to_datetime(times['DateTime']).dt.dayofweek

        #Add boolean for federal holiday
        times['Date'] = pd.to_datetime(times['DateTime']).dt.date
        times['Day'] = pd.to_datetime(times['DateTime']).dt.day
        times['Holiday'] = times['Date'].isin(calendar().holidays(start=datetime.date(2001, 1, 1), end=datetime.date(2022, 12, 31)).date)   
        times['Holiday'] = times.apply(lambda row: 1 if row['Holiday'] == True else 0, axis=1)
        
        #Account for holidays that take place on the weekend
        for i, row in times.iterrows():
            holidays = [[12, 25], [12, 31], [7, 4], [1,1], [10, 31], [3,17]]
            for pair in holidays:
                if(row['Month'] == pair[0] and row['Day'] == pair[1]):
                    times.loc[i, 'Holiday'] = 1
        
        #Save it out
        times.to_csv('sports_weather_hourly.csv')

    else:
        times = pd.read_csv('sports_weather_hourly.csv')

    crimes_keys = ['Year','Month','Day','Hour']
    times_keys = ['Year','Month','Day','HourGroup']

    crimes = crimes.merge(times, how="left", left_on=crimes_keys, right_on=times_keys)


    #Drop columns
    crimes = crimes[['Date_x','Year','Primary Type','Description','Latitude','Longitude','Arrest','Domestic','Community Area','Beat','Month','Hour',
                     'C','m/s','PRCP','SNOW','BullsGame','CubsGame','SoxGame','BearsGame','Weekday','Holiday']]

    crimes.columns = ['DateTime','Year','Primary Type','Description','Latitude','Longitude','Arrest','Domestic','Community Area','Beat','Month','Hour',
                     'C','m/s','PRCP','SNOW','BullsGame','CubsGame','SoxGame','BearsGame','Weekday','Holiday']

    crimes.sort_values(by='DateTime', inplace=True)
    crimes.dropna(inplace=True)
    crimes.reset_index(inplace=True, drop=True)

    #Save to csv for later use in model training
    crimes.to_csv('crimes_ext.csv')
    
    
    return crimes

def geocode_dest(dest):
  """Find the latitude and longitude of input destination/address
  Parameters
  ----------
  dest: input destination/address
    
  Returns
  -------
  Latitude and longitude of input destination/address. 
  """
  g = geocoder.osm(dest)
  if g.ok:
    result = g.json
    return result['lat'], result['lng']
  else:
    return None, None

def geocode_geojson(lat, lng, boundaries):
  """Find the Chicago Community Area for a given input latitude and longitude
  Parameters
  ----------
  lat: input latitude
  lng: input longitude
  geojson_file: geospatial data with boundaries e.g., Boundaries - Community Areas (current).geojson
    
  Returns
  -------
  Community Area of input latitude and longitude. None if not found e.g., outside of Chicago.
  """
  if (lat is not None) and (lng is not None):
    for feature in boundaries['features']:
      polygon = shape(feature['geometry'])
      if polygon.contains(Point(lng, lat)):
        return feature['properties']['area_num_1']
  else:
    return None

