########################################
#Code to pull & format sports schedules#
'''
Functions to take in and format raw sports data for combination with crime + weather
'''
###Library Imports###

#Pandas - dataframes & associated functions
import pandas as pd

#Requests - pulls data via urls
import requests

#Regular expressions
import re

#Handling of date & time values
import datetime

#OS
import os

#Numpy
import numpy as np


#What's the date?
today = datetime.date.today()
year = today.year
date = today.day
month = today.month

#Where is the raw data?
data_path = os.path.dirname(os.getcwd())

#Function to retrieve MLB schedules from CHC and CWS
def get_mlb_schedule(team_label):
  '''
  Function to generate the indicator schedule for specified Chicago MLB team

  Parameters
  ----------
  team_label: either 'CHC' or 'CWS' for Cubs and White Sox respectively.
  '''
  
  schedule_df = pd.DataFrame()

  #Pull schedule for each year starting in 2001
  for i in range(2001, year+1):
    pull = pd.read_html(f'https://www.baseball-reference.com/teams/{team_label}/{i}-schedule-scores.shtml#team_schedule')[0]
    pull['Year'] = i
    schedule_df = pd.concat([schedule_df, pull], axis=0)

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

  return schedule_df


def get_mlb_schedules():
  '''
  Function to combine the two MLB teams' schedules

  Paramters
  ---------
  None

  Returns
  -------
  DataFrame containing both MLB teams' schedules
  '''
  
  #Pull each team's schedules
  cubs_schedule = get_mlb_schedule('CHC')
  white_sox_schedule = get_mlb_schedule('CHW')

  #Combine for MLB teams
  mlb_schedule = pd.concat([white_sox_schedule, cubs_schedule], axis=0).sort_values(by='Date',ascending=True).reset_index(drop=True)
  return mlb_schedule


def get_nfl_schedule():
  '''
  Function to generate Chicago Bears' schedule

  Parameters
  ----------
  None

  Returns
  -------
  DataFrame containing Bears' schedule
  '''

  #Read in raw data
  schedule_df = pd.read_excel(os.path.join(data_path,'ChicagoNFL_raw.xlsx'))

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

  return schedule_df

def get_nba_schedule():
  '''
  Function to generate Chicago Bulls' schedule

  Parameters
  ----------
  None

  Returns
  -------
  DataFrame containing Bulls' schedule
  '''

  #Read in raw data
  schedule_df = pd.read_excel(os.path.join(data_path,'ChicagoNBA_raw.xlsx'))

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

  #Median length of NFL games are 2 hours and 12 minutes, set since we don't have duration of games
  schedule_df = schedule_df.assign(Duration=122)
  

  #Keep only these columns
  schedule_df = schedule_df[['Date','StartTime','Duration']]
  schedule_df = schedule_df.assign(Team='BULLS')

  return schedule_df

#String functions together to generate full schedule of all sports
sports_schedules = pd.concat([get_mlb_schedules(), get_nba_schedule(), get_nfl_schedule()], axis=0).sort_values(by='Date',
                ascending=True).reset_index(drop=True)

#Save to csv
sports_schedules.to_csv(os.path.join(data_path,'chicago_sports_clean.csv'))
