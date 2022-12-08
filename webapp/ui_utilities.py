import sys 
sys.path.append('..')

import pandas as pd
import datetime
from datetime import timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import matplotlib.pyplot as plt
from prophet.serialize import model_from_json
import pytz
import math
import time

plt.switch_backend('agg')

def is_holiday(datetime_input):
    is_holiday= False

    format_date_str = datetime_input.strftime('%Y-%m-%d')
    holidays =  calendar().holidays(start=datetime.date(2022,11, 14), end=datetime.date(2023, 12, 31))

    for day in holidays:
        holiday_date_str = day.strftime('%Y-%m-%d')
        if holiday_date_str == format_date_str:
            is_holiday= True

    #some holiday maybe on the weekend, the calendar().holidays only return the observed date
    more_holiday_list = [ '2022-12-25', '2023-01-01','2023-11-11']
    if format_date_str in more_holiday_list:
        is_holiday = True
        
    return is_holiday


def is_day(datetime_input):
    is_day = False
    if datetime_input.hour > 5 and datetime_input.hour < 18:
        is_day = True

    return is_day

def createClassifierGraph (url, result_dict):

    x_axis = []
    y_axis = []

    for key , value in result_dict.items():
        x_axis.append(key)
        y_axis.append(value)

    x_axis.reverse()
    y_axis.reverse()

    fig, ax = plt.subplots()

    ax.barh(x_axis, y_axis, color='blue')

    bars = ax.barh(x_axis, y_axis)
    ax.bar_label(bars)

    ax.set_xlabel('Scores')
    ax.set_ylabel('Crime Type')
    ax.set_title('Most probable crime type(s)')

    plt.xlim([0, .6])
    plt.savefig(url,bbox_inches="tight")

    plt.close('all')

def runTimeSeriesModel (fn , model_files_df, url,datetime_input,future_data ):
    with open(f'./static/model/time_series/{fn}.json', 'r') as fin:
        m = model_from_json(fin.read())  

    t_res = fn.split('_')[-1]
    ca = int(fn.split('_')[0].split('CA')[-1])

    model_file = model_files_df[model_files_df['ca']==ca]
    scaler = model_file['scaler'].iloc[0]
    train_exog = model_file['train_exog'].iloc[0]

    train_time = pytz.timezone('America/Chicago').localize(train_exog.iloc[-1]['ds'])

    bulls = []
    cubs = []
    sox = []
    bear = []
    temp = []
    wind =  []
    rain = []
    snow = []

    next_period = train_time + timedelta(hours=8)

    forecast_periods = 0 
    while next_period < datetime_input:
        df_lookup = future_data.query(f'Year=={next_period.year} and Month == {next_period.month} and Day == {next_period.day} and HourGroup >= {next_period.hour} and HourGroup < {next_period.hour + 8}')
  
        bulls.append(math.ceil(df_lookup['BullsGame'].mean()))
        cubs.append(math.ceil(df_lookup['CubsGame'].mean()) )
        sox.append(math.ceil(df_lookup['SoxGame'].mean()) )
        bear.append(math.ceil(df_lookup['BearsGame'].mean()) )
        temp.append(df_lookup['C'].mean() )
        wind.append(df_lookup['m/s'].mean() )
        rain.append(df_lookup['PRCP'].mean() )
        snow.append(df_lookup['SNOW'].mean() )

        forecast_periods = forecast_periods + 1
        next_period = next_period + timedelta(hours=8)

    future = m.make_future_dataframe(periods=forecast_periods, freq=t_res)
    future_rows = future.merge(train_exog['ds'], how='left', indicator=True)
    future_rows = future_rows[future_rows['_merge'] == 'left_only']
    future_exog = future_rows[['ds']]

    # add the future exog
    ##################################
    future_exog['bulls'] = bulls
    future_exog['cubs'] = cubs
    future_exog['sox'] = sox
    future_exog['bears'] = bear
    future_exog['temp'] = temp
    future_exog['wind'] = wind
    future_exog['rain'] = rain
    future_exog['snow'] = snow
    ##################################

    cols2scale = ['temp', 'wind', 'rain', 'snow']
    future_exog[cols2scale] = scaler.transform(future_exog[cols2scale])
    exog_all = pd.concat([train_exog, future_exog])
    future = future.merge(exog_all, how='left', on='ds')


    start = time.time()
    forecast = m.predict(future)
    end = time.time()

    print ("Time Series forcase takes :" , end - start , " seconds")
    m.plot(forecast).savefig(url)

    #forecast.to_csv("forecase.csv")

    return forecast

def calSafetyScore (crimes):
    score = 0

    # percentile99
    max_crime = 25

    if crimes > max_crime :
        return score

    else:
        return round((max_crime - crimes ) / max_crime * 100 )
