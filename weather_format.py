####################################
#Code to pull & format weather data#
####################################


###Library Imports###

#Pandas - dataframes & associated functions
import pandas as pd

#Requests - pulls data via urls
import requests

#Regular expressions
import re

#Handling of date & time values
import datetime

#Math operations
import numpy as np

#Plotting library
import matplotlib.pyplot as plt

#OS
import os


#Function to pull historical NOAA data to pandas dataframe
def pull_noaa_hist(year: int):

  """
  Pulls historical weather data from the NOAA database
  Parameters
  ----------
  year: int year of interest
    
  Returns
  -------
  DataFrame with columns for date, time, and weather conditions
  """

  #Error handling for years out of range
  if(year < 2000):
    return pd.DataFrame({})

  #Pull the txt table from the NOAA website
  response = requests.get(f'https://www.glerl.noaa.gov/metdata/chi/archive/chi{year}.04t.txt').text

  #Format it based on each row
  response = response.split("\n")[1:-1]

  #Define the headers by formatting the first item in the split list and reserve the rest as data
  col_headers = re.sub(' +', ' ',response[0]).strip().split(' ')
  
  #Change duplicate column name
  col_headers[6]='m/s_G'
  
  #For tables with relative humidity (starts in 2016), label the column
  if len(col_headers) == 9:
    col_headers[8] = 'RelHum'
  response = response[1:]

  #Format each line of data by splitting out into list items
  temp_response = []
  for i in range(len(response)):
    #Accounts for repeating headers through the text table
    if(response[i][1] == "4"):
      temp_response.append(re.sub(' +', ' ',response[i]).strip().split(' '))
  
  #Convert to numeric data frame, limiting formatting errors
  data = pd.DataFrame(temp_response).apply(pd.to_numeric)
  data = data.iloc[:,0:len(col_headers)]
  data.columns = col_headers
  data.drop(data.columns[[0,6,7]], axis=1, inplace=True)

  return data

#Function to get current data from NOAA
#Define a start date to stop at to not pull the full year every time, just what isn't already in the file
def pull_noaa_current():

  """
  Pulls weather data in current year from NOAA database
  Parameters
  ----------
  
    
  Returns
  -------
  DataFrame with columns for date, time, and weather conditions
  """

  #Grab today's date
  today = datetime.date.today()
  year = today.year
  date = today.day
  month = today.month

  #Make sure the date is formatted corretly to pipe into the link
  day = today.day
  if len(str(day)) == 1:
    day = '0'+str(day)

  month = today.month
  if len(str(month)) == 1:
    month = '0'+str(month)

  #Pull the txt table from the NOAA website; start with the first day of the year
  response = requests.get(f'https://www.glerl.noaa.gov/metdata/chi/{year}/{year}{month}{day}.04t.txt').text

  #Format it based on each row
  response = response.split("\n")[1:-1]

  #Define the headers by formatting the first item in the split list and reserve the rest as data
  col_headers = re.sub(' +', ' ',response[0]).strip().split(' ')
  
  #Change duplicate column name
  col_headers[6]='m/s_G'
  
  #For tables with relative humidity (starts in 2016), label the column
  if len(col_headers) == 9:
    col_headers[8] = 'RelHum'
  response = response[1:]

  #Format each line of data by splitting out into list items
  temp_response = []
  for i in range(len(response)):
    #Accounts for repeating headers through the text table
    if(response[i][1] == "4"):
      temp_response.append(re.sub(' +', ' ',response[i]).strip().split(' '))
    
  #Convert to numeric data frame, limiting formatting errors
  data = pd.DataFrame(temp_response).apply(pd.to_numeric)
  data = data.iloc[:,0:len(col_headers)]
  data.columns = col_headers
  data.drop(data.columns[[0,6,7]], axis=1, inplace=True)

  #Now repeat the process, going back until the first day of the year
  date = today - datetime.timedelta(days=1)

  while date.year == year:
    day = date.day
    if len(str(day)) == 1:
      day = '0'+str(day)
    
    month = date.month
    if len(str(month)) == 1:
      month = '0'+str(month)
    response = requests.get(f'https://www.glerl.noaa.gov/metdata/chi/{year}/{year}{month}{day}.04t.txt').text

    response = response.split("\n")[1:-1]

    col_headers = re.sub(' +', ' ',response[0]).strip().split(' ')
    col_headers[6]='m/s_G'
    
    if len(col_headers) == 9:
      col_headers[8] = 'RelHum'
    response = response[1:]

    temp_response = []
    for i in range(len(response)):
      if(response[i][1] == "4"):
        temp_response.append(re.sub(' +', ' ',response[i]).strip().split(' '))

    temp_data = pd.DataFrame(temp_response).apply(pd.to_numeric)
    temp_data = temp_data.iloc[:,0:len(col_headers)]
    temp_data.columns = col_headers
    temp_data.drop(temp_data.columns[[0,6,7]], axis=1, inplace=True)

    data = pd.concat([data, temp_data], axis=0)

    date -= datetime.timedelta(days=1)
  
  return data.reset_index(drop=True)

def compile_weather(file_name, delta):
  """
  Combines historical weather data with current year data and reformats
  Parameters
  ----------
  file_name: file name of historical weather data from pull_noaa_hist()
  delta: time difference desired between each observation
    
  Returns
  -------
  DataFrame with columns for date, time, and weather conditions to match up with cleaned crimes dataset
  """
  
  #Don't go back and pull old historical data if we already have it
  historical_weather_path = file_name

  historical_weather = pd.DataFrame({})

  if os.path.isfile(historical_weather_path):
    historical_weather = pd.read_csv(historical_weather_path)
    historical_weather.drop(historical_weather.columns[[0]], axis=1, inplace=True)

  else:
    historical_weather = pd.DataFrame()
    for i in range(2001, datetime.date.today().year):
      new_data = pull_noaa_hist(i)
      historical_weather = pd.concat([historical_weather, new_data])

  historical_weather = pd.concat([historical_weather, pull_noaa_current()], axis=0)

  #Keep just the records that meet the time scale we are looking at; 100 = 1 hour,
  def strip_data(delta, historical_weather):
    
    historical_weather = historical_weather[historical_weather['UTC'] % delta == 0]
    return historical_weather

  historical_weather = strip_data(delta, historical_weather)

  #Convert UTC to time
  historical_weather['Time'] = historical_weather.apply(lambda row: str(int(row['UTC']/100))+":00", axis=1)
  historical_weather['Time'] = historical_weather.apply(lambda row: datetime.datetime.strptime(row['Time'], "%H:%M").strftime('%I:%M %p'), axis=1)


  #Add column to historical weather data for date to merge with rain & snow
  historical_weather['Date'] = historical_weather.apply(lambda row: datetime.datetime.strptime(f"{int(row['Year'])} {int(row['DOY'])}", '%Y %j'), axis=1)


  #Combine Date & Time
  historical_weather['DateTime'] = historical_weather.apply(lambda row: datetime.datetime.combine(row['Date'], datetime.datetime.strptime(str(row['Time']), "%I:%M %p").time()), axis=1)
  historical_weather['DateTime'] = pd.to_datetime(historical_weather['DateTime'])

  #Add Relevant Variables for time
  historical_weather['Year'] = historical_weather['DateTime'].dt.strftime("%Y").astype(int)
  historical_weather['Month'] = historical_weather['DateTime'].dt.strftime("%m").astype(int)
  historical_weather['Day'] = historical_weather['DateTime'].dt.strftime("%d").astype(int)
  historical_weather['HourGroup'] = historical_weather['DateTime'].dt.strftime("%H").astype(int)
  historical_weather['Date'] = historical_weather['DateTime'].dt.strftime("%M/%D/%Y")

  #Drop out all the extra columns
  historical_weather.drop(columns=['DOY','UTC','DATE','Date', 'DateTime', 'Time'],inplace=True)
  historical_weather.drop_duplicates(inplace=True)

  #Merge with precipitation data
  rain_snow = pd.read_csv('rain_snow_raw.csv')
  rain_snow['DateTime'] = pd.to_datetime(rain_snow['DATE'])
  
  rain_snow['Month'] = rain_snow['DateTime'].dt.strftime("%m").astype(int)
  rain_snow['Day'] = rain_snow['DateTime'].dt.strftime("%d").astype(int)
  rain_snow['Year'] = rain_snow['DateTime'].dt.strftime("%Y").astype(int)



  left_on = ['Year','Month','Day']
  right_on = ['Year','Month','Day']
  historical_weather = historical_weather.merge(rain_snow, left_on=left_on, right_on=right_on, how='left')

  #Drop further columns from merge
  historical_weather.drop(columns=['STATION','NAME','DATE','DateTime'],inplace=True)
  historical_weather.drop_duplicates(inplace=True)
  historical_weather.reset_index(drop=True)
  
  #Save out to csv!
  delta = delta//100
  historical_weather.to_csv(f'historical_weather_clean_{delta}h.csv')
  return historical_weather
