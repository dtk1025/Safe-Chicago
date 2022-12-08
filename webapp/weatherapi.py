import urllib.request
import urllib.error
import datetime
import sys
import json
import pytz
from datetime import timedelta
import logging

class Weather:
    temp = 0
    windspeed = 0
    rain = 0
    snow = 0


def getWeather_14day(weather, input_datetime):
    current_epoch  = int(round(input_datetime.timestamp()))
    
    URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/chicago?unitGroup=metric&include=hours%2Cdays&key=PK9P2KE4RNDQ96TL49EQJ4DD6&contentType=json"
   
    try: 
        data = urllib.request.urlopen(URL, timeout = 3)
        logging.info (str(data.getcode()))
        if (data.getcode() == 200) :
            weatherData = json.loads(data.read().decode('utf-8')) 
            day_list = weatherData["days"]

            for day in day_list:
                hour_list = day["hours"]
                day_epoch = day["datetimeEpoch"]
                if (day_epoch <= current_epoch):
                    rain_temp = day.get('precip','default value')
                    snow_temp = day.get('snow','default value')
                    if rain_temp is not None:
                        weather.rain = rain_temp
                    if snow_temp is not None:
                        weather.snow = snow_temp

                for hour in hour_list:
                    hour_epoch = hour["datetimeEpoch"]
                    if (hour_epoch <= current_epoch):
                        weather.temp = hour["temp"]
                        weather.wind_speed = hour["windspeed"]
                    else:
                        break
            
    
        else:
            logging.info("Weather API is not working, use history look up")

    except urllib.error.HTTPError  as e:
        logging.info(e)
        logging.info("Weather API is not working, use history look up")
        #sys.exit()
    except  urllib.error.URLError as e:
        logging.info(e)
        logging.info("Weather API is not working, use history look up")
        #sys.exit()