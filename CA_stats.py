import pandas as pd
import numpy as np


#Define df to hold all stats
stats_df = pd.DataFrame({})

for i in range(0, 78):
    if i == 0:
        data = pd.read_csv('crimes_agg_8h.csv')

    else:
        data = pd.read_csv(f'crimes_agg_ext_beat{i}_8h.csv')

    #data = data[data['Year'] >= 2017]

    counts = data['Count']

    mean = np.mean(counts)
    median = np.median(counts)
    sd = np.std(counts)
    minimum = min(counts)
    maximum = max(counts)

    #percentile_n_list = [i for i in range(0, 101)]
    percentile_list = [np.percentile(counts, i) for i in range(0, 101)]

    percentile_col_list = [f'percentile{i}' for i in range(0, 101)]
    col_list = ['CA','mean','median','std dev','min','max']
    col_list.extend(percentile_col_list)

    stat_list = [i,mean,median,sd,minimum,maximum]
    stat_list.extend(percentile_list)

    stats_df[i]=stat_list


stats_df = stats_df.T
stats_df.columns = col_list

stats_df.to_csv('stats_df.csv')

#Also do chicago on the whole

    
    
