import sys 
sys.path.append('..')

from flask import Flask,flash,request, render_template
import pickle
import datetime
import logging
import googlemaps
import pytz
from weatherapi import *
from ui_utilities import *
import pandas as pd
import utilities
import pygeohash as pgh
import random
import time

app = Flask(__name__)

# set up logging
logging.basicConfig(filename="safe_chicago.log", level=logging.INFO)

#load future data
future_data = pd.read_csv('./static/data/future_data_full.csv')

#load CA avg crime
crime_avg_data = pd.read_csv('./static/data/stats_df_5yr.csv')

# load and setup the classifier model
fn = './static/model/classifier/VotingClassifier1_save_file.pkl'
with open(fn, 'rb') as f:
    clf_save_file = pickle.load(f)

eclf = clf_save_file['clf']
le = clf_save_file['le']
preprocessor = clf_save_file['preprocessor']

#load the time series model 
model_files_df = pd.read_pickle('./static/model/time_series/prophet_model_files.pkl')

@app.route('/')
def home():
    return render_template('index.html' , grapsh_display= f'none')

@app.route('/predict',methods=['POST'])
def predict():

    # Get the location and datetime from the user inputs
    attractions = str(request.form["attractions"])
    address = str(request.form["destination"])
    date_input_str = request.form["date"]
    time_input_str = request.form["time"]

    # Get the destination based on attraction or address
    destination = ""
    if attractions:
        destination = attractions
    elif address :
        destination = address
    else:
        return render_template('index.html', error_text = f' Address is empty, please select an attraction or enter a valid Chicago address', grapsh_display= f'none')

    # Convert the user inputs of date and time to datetime object (America/Chicago timezon)
    datetime_input_str = date_input_str + " "+ time_input_str
    datetime_input = datetime.datetime.strptime(datetime_input_str, '%Y-%m-%d %H:%M')
    datetime_input = pytz.timezone('America/Chicago').localize(datetime_input)
    datetime_input_epoch = int(round(datetime_input.timestamp()))

    print ("input time: " , datetime_input, " , epoch: " , datetime_input_epoch )
    
    # convert address to geographic coordinates
    gmaps = googlemaps.Client(key='AIzaSyBABPkwvPefO3l8eURiIRF0TfI2wAAndLI')
    geocode_result = gmaps.geocode(destination)

    if len(geocode_result) == 0:
        return render_template('index.html', error_text = f' Invalid address [{destination}], please enter a valid Chicago address', grapsh_display= f'none')

    formated_address = geocode_result[0]["formatted_address"]
    address_without_street = formated_address.split ( "," ,1)[1]

    # Check if the input address contaisn Chicago as a city
    if "chicago" not in address_without_street.lower() :
        return render_template('index.html', error_text = f' Address [{destination}] is not in Chicago , please enter a valid Chicago address' , grapsh_display= f'none')

    # Get the lat and lng
    lat = geocode_result[0]["geometry"]["location"]["lat"]
    lng = geocode_result[0]["geometry"]["location"]["lng"]

    model_input_str = " ------ Prediction inputs: destination[" + destination + "][" + str(lat) + " , " + str(lng) + "], date[" + date_input_str + "], time[" + time_input_str +"]------"
    logging.info(model_input_str)
    print(model_input_str)

    df_lookup = future_data.query(f'Year=={datetime_input.year} and Month == {datetime_input.month} and Day == {datetime_input.day} and HourGroup == {datetime_input.hour}')
    print (df_lookup)

    # get the weather info 
    weather = Weather()
    buls_input = False
    cubs_input = False
    sox_input = False
    bears_input = False

    if len(df_lookup) >= 1:
        weather.temp = df_lookup.iloc[0]['C']
        weather.windspeed =  df_lookup.iloc[0]['m/s']
        weather.rain =  df_lookup.iloc[0]['PRCP']
        weather.snow =  df_lookup.iloc[0]['SNOW']
        if df_lookup.iloc[0]['BullsGame'] == 1 :
            buls_input = True
        if df_lookup.iloc[0]['BearsGame'] == 1 :
            bears_input = True
        if df_lookup.iloc[0]['CubsGame'] == 1 :
            cubs_input = True
        if df_lookup.iloc[0]['SoxGame'] == 1 :
            sox_input = True

    now = datetime.datetime.now(pytz.timezone('America/Chicago'))

    if datetime_input < now + timedelta(days=14):
        logging.info ("less than 14 days, calling api to get weather info")
        print("less than 14 days, calling api to get weather info")
        getWeather_14day(weather, datetime_input)


    gh = pgh.encode(lat, lng, precision=6)
  
    fn = '../Boundaries - Police Beats (current).geojson'
    with open(fn) as fh:
        boundaries = json.load(fh)

    beat_input = utilities.geocode_geojson_beat(lat=lat, lng=lng, boundaries=boundaries)

    fn = '../Boundaries - Community Areas (current).geojson'
    with open(fn) as fh:
        boundaries = json.load(fh)

    community_area_input = utilities.geocode_geojson(lat=lat, lng=lng, boundaries=boundaries)

    # populating the inputs for the models
    teap_input = weather.temp
    wind_input = weather.windspeed
    rain_input = weather.rain
    snow_input = weather.snow
    isholiday_input = is_holiday(datetime_input)
    month_input = datetime_input.month
    day_input = datetime_input.day
    hour_input = datetime_input.hour
    dayofweek_input = datetime_input.weekday()
    isday_input = is_day(datetime_input)
    geohash_input = gh

    # construct the df input for the model
    from_user = pd.DataFrame({'beat':beat_input, 'temp':teap_input, 'wind':wind_input, 'rain':rain_input, 'snow':snow_input,
                          'bulls':buls_input, 'cubs':cubs_input, 'sox':sox_input, 'bears':bears_input,
                          'isholiday':isholiday_input, 'month':month_input, 'day':day_input, 'hour':hour_input, 'dayofweek':dayofweek_input, 'isday':isday_input, 'geohash':geohash_input},
                         index=[0])

   
    # Predict with the classifier model

    logging.info(from_user)
    print(from_user)

    start = time.time()

    to_display = utilities.get_crime_types(clf=eclf, le=le, X_test_sample=pd.DataFrame(preprocessor.transform(from_user), columns=preprocessor.get_feature_names_out()))
    end = time.time()
    print ("Classifier prediction takes :" , end - start , " seconds")

    logging.info(to_display)
    print(to_display)

    to_display_formatted = {}

    for key  in to_display:
        new_key = key.title()
        to_display_formatted[new_key] = round (to_display[key],3)

    c_url = "./static/image/classifier/c-" + str(random.randint(0, 100000)) + ".PNG"
    createClassifierGraph(c_url, to_display_formatted)

    # Predict with the time series model
    print ("community_area_input: ", community_area_input)
    t_url = "./static/image/time_series/t.PNG" 
    file_name = 'CA' + str(community_area_input) + '_8H' # model file name to load
    forecast = runTimeSeriesModel(file_name,model_files_df,t_url,datetime_input, future_data)

    predict_date = pytz.timezone('America/Chicago').localize(forecast["ds"].iloc[-1])
    predict_score = round(forecast["yhat"].iloc[-1], 2)

    crime_avg = crime_avg_data.query(f'CA=={community_area_input}')
    y_hat_avg = crime_avg.iloc[0]['mean']
    print ('yhat average: ', y_hat_avg)

    ca_average = calSafetyScore(y_hat_avg)
    safety_score = calSafetyScore(predict_score)

    # Return the results to the UI
    return render_template('index.html', 
    user_input_text = f'{destination} (CA{community_area_input}):',
    # model_input_text = f'Address: {destination}',
    #model_input_text = f'Classifier Model Inputs: beat[{beat_input}], temp[{teap_input}], wind[{wind_input}], rain[{rain_input}], snow[{snow_input}], bulls[{buls_input}], cubs[{cubs_input}], sox[{sox_input}], bears[{bears_input}], isholiday[{isholiday_input}], month[{month_input}], day[{day_input}], hour[{hour_input}], dayofweek[{dayofweek_input}], isday[{isday_input}], geohash[{geohash_input}]',
    t_info_text = f'Safety Score (5-year average): {ca_average}',
    t_predit=f'Safety Score ({datetime_input_str}): {safety_score} ',
    safety_score = f'{safety_score}',
    # c_output_text=f'Classifier Model Outputs:',
    c_url = f'{c_url}',
    t_output_text=f'Probable number of crime(s) in 8 hours: {predict_score}',
    
    #t_url = f'{t_url}',
    grapsh_display= f'block'
    #safety_level = random.randint(1,5)
    )

if __name__ == "__main__":
    print ("running on: http://192.168.7.72:8080")
    app.run(host="0.0.0.0", port=8080)
    