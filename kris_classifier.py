#note to look at feature importances


import os
import pandas as pd
import numpy as np
import datetime
import dateutil.parser
import matplotlib.pyplot as plot
import time
from load_data import *
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
import sklearn.linear_model
import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE as smote
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import random
import seaborn as sns




import warnings
warnings.filterwarnings("ignore")


def prepare_data(data, CA_num, mapping_dict):
    '''
    filepath: cleaned crimes_ext.csv file location
    CA_num: number of community area
    '''

    '''
    data = pd.DataFrame({})
    
    if(os.path.isfile(filepath)):
        data = pd.read_csv(filepath)
    else:
        data = merge_extra_features('crimes_clean.csv')
    '''

    data = data[data['Community Area'] == CA_num]

    #Drop columns that won't be of use
    drop_cols = ['DateTime','Description','Arrest','Domestic',
                 'Beat','Community Area','Year']
    data.drop(columns=drop_cols, inplace=True)

    #Drop columns without latitude values
    data = data[data['Latitude'].notna()]

    #Remove duplicates
    data.drop_duplicates(inplace=True)

    #Shuffle
    data = data.sample(frac=1)
    data.reset_index(inplace=True, drop=True)

    #Drop artifact columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    #Scale continuous variables
    for col in ['C','m/s','PRCP','SNOW', 'Latitude','Longitude']:
        scaler = StandardScaler()
        data[col] = scaler.fit_transform(np.array(data[col]).reshape(-1,1))

    #Create dummies for categorical variables
    cat_cols = ['Hour','Month','Weekday']
    data= pd.get_dummies(data, columns=cat_cols)

    #Map the crime types and fill the rest with 'Other'
    data['Primary Type'] = data['Primary Type'].map(mapping_dict)
    data['Primary Type'] = data['Primary Type'].fillna('Other (Disorderly conduct, public intoxication, etc.)')

    #Setup train-test split at 80% for training
    split_index = 4 * (data.shape[0]//5)
    train, test = data.iloc[0:split_index, :],data.iloc[split_index:, :]

    x_train = train.drop(columns='Primary Type')
    x_test = test.drop(columns='Primary Type')

    y_train = train['Primary Type']
    y_test = test['Primary Type']
    
    return x_train, x_test, y_train, y_test

def print_performance(true, pred, printout=True):

    performance = pd.DataFrame(sklearn.metrics.classification_report(true, pred, output_dict=True)).transpose().sort_values(by='support')
    
    acc = round(performance.iloc[0,0],3)

    imp_over_naive = round(acc - (performance.iloc[-3,3]/performance.iloc[-2,3]),3)
    
    if(printout):
        print("Full Report:\n")
        print(performance)

        print(f"\nOverall Accuracy: {acc}")
        print(f"Improvement over naive guessing: {imp_over_naive}")

    macro_prec = round(performance.iloc[-2,0],3)
    macro_rec = round(performance.iloc[-2,1],3)
    macro_f1 = round(performance.iloc[-2,1],3)

    weighted_prec = round(performance.iloc[-1,0],3)
    weighted_rec = round(performance.iloc[-1,1],3)
    weighted_f1 = round(performance.iloc[-1,1],3)

    return_list = [acc, imp_over_naive, macro_prec, macro_rec, macro_f1, weighted_prec, weighted_rec, weighted_f1]

    return return_list


def run_test(data, mapping_dict, beat_list = None, beat_num = 20):

    if beat_list == None:
        beat_list = random.sample([i for i in range(1, 78)], beat_num)
        
    performance_mat = []

    print("Start!")

    for i, beat in enumerate(beat_list):

        x_train, x_test, y_train, y_test = prepare_data(data, int(beat), mapping_dict)

        #Train random forest
        rf = RandomForestClassifier()

        print("cross validating...")
        cross_val = RepeatedStratifiedKFold(n_splits=5)

        param_grid =  {'n_estimators':[5, 10, 15],'criterion':['entropy']}
        

        search = GridSearchCV(rf, param_grid = param_grid, scoring='accuracy')
        search.fit(x_train, y_train)

        rf_best = search.best_estimator_

        rf_best.fit(x_train, y_train)

        #Make predictions based on the test set
        rf_preds = rf_best.predict(x_test)

        rf_perf = print_performance(y_test, rf_preds, False)

        performance_mat.append([beat] + rf_perf)

        print(f"Done with beat {beat}...({i+1}/{beat_num})")


    perf_df = pd.DataFrame(performance_mat)
    col_list = ['Beat','Overall Acc.','Improvement over Guess',
                'Macro Prec.','Macro Recall','Macro F1',
                'Weighted Prec.','Macro Recall','Macro F1']
    perf_df.columns = col_list
    perf_df.set_index('Beat',drop=True)

    print(perf_df)
    return perf_df
        

#Define mapping dictionary to combine certain high-frequency crimes.
mapping_dict = {'BATTERY' : 'Assault', 
              'THEFT': 'Theft',
              'NARCOTICS': 'Drug',
               'ASSAULT': 'Assault',
               'BURGLARY': 'Burglary',
               'ROBBERY': 'Robbery',
               'CRIMINAL DAMAGE': 'Property',
               'WEAPONS VIOLATION': 'Weapons',
               'DECEPTIVE PRACTICE': 'Fraud',
               'CRIMINAL TRESPASS': 'Property',
               'MOTOR VEHICLE THEFT': 'Theft',
               'SEX OFFENSE': 'Sex',
               'PROSTITUTION': 'Sex',
               'CRIM SEXUAL ASSAULT': 'Sex',
               'ARSON': 'Property',
               'CONCEALED CARRY LICENSE VIOLATION': 'Weapons',
              'CRIMINAL SEXUAL ASSAULT': 'Sex',
              'PUBLIC INDECENCY': 'Sex',
               'OTHER NARCOTIC VIOLATION': 'Drug',
                'KIDNAPPING': 'Kidnapping / Trafficking',
                'HUMAN TRAFFICKING': 'Kidnapping / Trafficking',
                'STALKING': 'Kidnapping / Trafficking',
                'HOMICIDE': 'Homicide'
              }





#Generate data to be used for classifier
data = pd.read_csv('crimes_ext.csv')
#x_train, x_test, y_train, y_test = prepare_data('crimes_ext.csv', 1, mapping_dict)

b = run_test(data, mapping_dict, beat_num = 20)
print(b['Improvement over Guess'].mean())
print(b['Overall Acc.'].mean())
b.to_csv('rf_testing_perf.csv')

#Setup a dataframe to store our performance
perf_mat = []

#############################
#       RF classifiers      #
#############################

'''
#Naive Random forest classifier#
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

#Make predictions based on the test set
rf_preds = rf.predict(x_test)

print("Random Forest, no oversampling, 100 trees")
rf_perf = print_performance(y_test, rf_preds)

perf_mat.append(['Random Forest (100 estimators)'] + rf_perf)

#Naive Random forest classifier, 500 estimators#
rf_500 = RandomForestClassifier(n_estimators=500)
rf_500.fit(x_train, y_train)


#Make predictions based on the test set
rf_500_preds = rf_500.predict(x_test)

print("\nRandom Forest, no oversampling, 500 trees")
rf_500_perf = print_performance(y_test, rf_500_preds)

perf_mat.append(['Random Forest (500 estimators)'] + rf_500_perf)

#Random forest classifier with systematic oversampling#

smote = smote()
x_train_os, y_train_os = smote.fit_resample(x_train, y_train)

rf_os = RandomForestClassifier()
rf_os.fit(x_train_os, y_train_os)

rf_os_preds = rf_os.predict(x_test)

print("Random Forest, oversampling, 100 trees")
rf_os_perf = print_performance(y_test, rf_os_preds)

perf_mat.append(['Random Forest + Oversample'] + rf_os_perf)

#Random forest with systematic oversampling and PCA#

'''
'''
#How many features do we really need?
pca = PCA()

x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

exp_var = pca.explained_variance_ratio_
cumulative_exp_var = np.cumsum(exp_var)
plot.step(range(0,len(cumulative_exp_var)), cumulative_exp_var)
plot.show(block=False)
'''
'''

#After 5 we start to slow down
pca = PCA(n_components=5)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)


x_train_pca_os, y_train_os = smote.fit_resample(x_train_pca, y_train)


rf_os_pca = RandomForestClassifier()
rf_os_pca.fit(x_train_pca_os, y_train_os)

rf_os_pca_preds = rf_os_pca.predict(x_test_pca)

print("Random Forest, oversampling, 100 trees, PCA reduced data")
rf_os_pca_perf = print_performance(y_test, rf_os_pca_preds)

perf_mat.append(['Random Forest + Oversample + PCA'] + rf_os_pca_perf)


######################
# K-Nearest Neighbors#
######################


# KNN No Oversampling #

knn_33 = KNeighborsClassifier(n_neighbors=33)
knn_33.fit(x_train, y_train)

knn_33_preds = knn_33.predict(x_test)

print("KNN, 33 neighbors")
knn_perf = print_performance(y_test, knn_33_preds)

perf_mat.append(['Default KNN (33 neighbor)'] + knn_perf)


# KNN SMOTE #

x_train_os, y_train_os = smote.fit_resample(x_train, y_train)

knn_33_os = KNeighborsClassifier(n_neighbors=33)
knn_33_os.fit(x_train_os, y_train_os)

knn_33_os_preds = knn_33_os.predict(x_test)

print("KNN, 33 neighbors, SMOTE oversampling")
knn_os_perf = print_performance(y_test, knn_33_os_preds)

perf_mat.append(['Oversample KNN (33 neighbor)'] + knn_os_perf)


############################
# Neural Network Classifier#
############################

#No oversampling#

#Add a new validation data set off the training set
split_index = 4 * (x_train.shape[0]//5)

x_train, x_val = x_train.iloc[0:split_index, :], x_train.iloc[split_index:, :]
y_train, y_val = y_train.iloc[0:split_index], y_train.iloc[split_index:]

#Format data to work with a NN
enc = LabelEncoder()

y_train_enc = enc.fit_transform(np.array(y_train).reshape(-1,1)).transpose()
y_val_enc = enc.fit_transform(np.array(y_val).reshape(-1,1)).transpose()

y_train_ohe = np_utils.to_categorical(y_train_enc)
y_val_ohe = np_utils.to_categorical(y_val_enc)

num_classes = y_train_ohe.shape[1]

#Define function to create a new model
def create_nn():
    nn = keras.models.Sequential()

    nn.add(keras.layers.Dense(32, activation='relu', input_dim=x_train.shape[1]))
    
    nn.add(keras.layers.Dense(64, activation='relu'))
    nn.add(keras.layers.Dropout(.2))
    
    nn.add(keras.layers.Dense(256, activation='relu'))
    nn.add(keras.layers.Dropout(.2))
    
    nn.add(keras.layers.Dense(1024, activation='relu'))
    #nn.add(keras.layers.Dropout(.2))
           
    nn.add(keras.layers.Dense(num_classes, activation='softmax'))

    nn.compile(loss='categorical_crossentropy',optimizer='sgd')

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)

    return nn, es


#Train
nn1, es = create_nn()
nn1.fit(x_train, y_train_ohe, epochs=100, validation_data=(x_val, y_val_ohe), callbacks=es, verbose=0)

nn1_preds = enc.inverse_transform(np.argmax(nn1.predict(x_test), axis=1))

print("Neural network, no oversampling")
nn1_perf = print_performance(y_test, nn1_preds)

perf_mat.append(['Default Neural Net'] + nn1_perf)


#Try oversampling
x_train_os, y_train_os = smote.fit_resample(x_train, y_train)
split_index = 4 * (x_train_os.shape[0]//5)

#Split into training/val
x_train_os, x_val_os = x_train_os.iloc[0:split_index, :], x_train_os.iloc[split_index:, :]
y_train_os, y_val_os = y_train_os.iloc[0:split_index], y_train_os.iloc[split_index:]

#Re-encode
enc = LabelEncoder()

y_train_os_enc = enc.fit_transform(np.array(y_train_os).reshape(-1,1)).transpose()
y_val_os_enc = enc.transform(np.array(y_val_os).reshape(-1,1)).transpose()

y_train_os_ohe = np_utils.to_categorical(y_train_os_enc)
y_val_os_ohe = np_utils.to_categorical(y_val_os_enc)

nn2, es = create_nn()
nn2.fit(x_train_os, y_train_os_ohe, epochs=1, validation_data=(x_val_os, y_val_os_ohe), callbacks=es, verbose=0)

nn2_preds = enc.inverse_transform(np.argmax(nn2.predict(x_test), axis=1))

print("Neural network, SMOTE oversampling")
nn2_perf = print_performance(y_test, nn2_preds)

perf_mat.append(['Oversample Neural Net'] + nn2_perf)

perf_df = pd.DataFrame(perf_mat)

col_list = ['Model Desc.','Overall Acc.','Improvement over Guess',
            'Macro Prec.','Macro Recall','Macro F1',
            'Weighted Prec.','Macro Recall','Macro F1']
perf_df.columns = col_list
perf_df.set_index('Model Desc.', inplace=True)



print(perf_df)

perf_df.to_csv('kris_classifier_perf.csv')
'''


