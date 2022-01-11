# Case 5 BIS - Lucas Winkler
# import libaries
import pandas as pd
import numpy as np
import traceback, sys
import os
from matplotlib import pyplot as plt


def raw_data():
    path = 'C:/Users/Lucas/Desktop/Master/01_Semester/Betriebliche Informationssysteme/Case 5 - FlyUIBK/raw_data.csv'
    try:
        normPath = path.replace(os.sep, '/')
    except:
        print('Error while normalizing path.')
    df = pd.read_csv(normPath, sep=';')
    df.fillna(0)
    df = df.astype({"Arrival delay in minutes":'int64', "Day of Week":'int64',"Delay indicator":'int64'})
    return df

def delay_plots(df):
    #print(df.dtypes)
    # plot occurences of delays per day of week for FlyUIBK
    temp_df = df.loc[df['Delay indicator'] == 1]
    temp_df = temp_df.loc[df['Airline']== 'FlyUIBK']
    temp_df = (temp_df.groupby(['Arrival delay in minutes','Day of Week']).size())
    #temp_df.groupby('Day of Week').count().plot(x='Day of Week', y='Arrival delay in minutes', kind='bar')
    #plt.show()
    # plot occurences of delays per day of week for FlyUIBK
    temp2_df = df.loc[df['Delay indicator']==1]
    temp2_df = temp2_df.loc[temp2_df['Airline'].str.contains('LDA')]
    temp2_df = (temp2_df.groupby(['Arrival delay in minutes','Day of Week']).size())
    #temp2_df.groupby('Day of Week').count().plot(x='Day of Week', y='Arrival delay in minutes', kind='bar')
    #plt.show()
    return temp_df, temp2_df

def statistics(df):
    df['Delay_per_minute_flight_duration'] = df['Arrival delay in minutes']/df['expec_flight_duration']
    df.to_csv('C:/Users/Lucas/PycharmProjects/bis_case5/output.csv',
              sep=';')
    # plot distribution of delays per airline
    temp_df = df.loc[df['Airline'] == 'FlyUIBK']
    temp_df['Arrival delay in minutes'].value_counts().sort_index().plot(kind='bar')
    plt.axvline(x=15, color='#DC143C')
    plt.show()
    temp2_df = df.loc[df['Airline'].str.contains('LDA')]
    temp2_df['Arrival delay in minutes'].value_counts().sort_index().plot(kind='bar')
    plt.axvline(x=15, color='#DC143C')
    plt.show()
    print('Mean delay of FlyUIBK in minutes:', round(temp_df['Arrival delay in minutes'].mean(),2))
    print('Mean delay of LDA in minutes:', round(temp2_df['Arrival delay in minutes'].mean(),2))
    print('Median delay of FlyUIBK in minutes:', round(temp_df['Arrival delay in minutes'].median(), 2))
    print('Median delay of LDA in minutes:', round(temp2_df['Arrival delay in minutes'].median(), 2))
    # how to normalize differences in estimated flight durations? @stefan
    # possible solution: delay per minute fly time --> normalized count of flights + delay
    # print('Mean delay per minute planned flight duration (FlyUIBK):', round(temp_df['Delay_per_minute_flight_duration'].mean(),2))
    # print('Mean delay per minute planned flight duration (LDA):',
    #      round(temp2_df['Delay_per_minute_flight_duration'].mean(), 2))
    return

def hypothesis():
    #hypothesis test to find out whether LDA truly performs better than FlyUBIK regarding the delay times
    # wilcoxon rank sum test seems like a good solution to test whether flyuibk average delay is higher than LDA's
    return

# open ideas:
"""
    - after clarifying whether FlyUIBK performs worse than LDA it might be helpful to find reasons for delay in 
      dataset --> e.g. Regression analysis
    - maybe get those statistics per route and airline --> just a change in input params @main function
"""

def normalizing_duration(df):
    # not resistant to flights with different dates (e.g. flight departures at 11pm and arrives at 2am the next day)
    # since there is no such case in the dataset it is neglected
    df['Departure'] = pd.to_datetime(df['Departure date'] + ' ' + df['Scheduled departure time'])
    df['expec_arrival'] = pd.to_datetime(df['Departure date'] + ' ' + df['Scheduled arrival time'])
    df['actual_arrival'] = pd.to_datetime(df['Departure date'] + ' ' + df['Actual arrival time'])
    df['expec_flight_duration'] = df['expec_arrival'] - df['Departure']
    df['expec_flight_duration'] = df['expec_flight_duration']/np.timedelta64(1,'m')
    df['actual_flight_duration'] = df['actual_arrival'] - df['Departure']
    df['actual_flight_duration'] = df['actual_flight_duration']/np.timedelta64(1,'m')
    df.to_csv('C:/Users/Lucas/PycharmProjects/bis_case5/output.csv', sep=';')
    return df

def define_routes(df):
    # save routes in new dataframe
    txl_vie = df[(df['Origin airport']=='TXL') & (df['Destination airport']=='VIE')]
    vie_txl = df[(df['Origin airport']=='VIE') & (df['Destination airport']=='TXL')]
    vie_osl = df[(df['Origin airport']=='VIE') & (df['Destination airport']=='OSL')]
    osl_vie = df[(df['Origin airport']=='OSL') & (df['Destination airport']=='VIE')]
    return txl_vie, vie_txl, vie_osl, osl_vie

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = raw_data()
    df = normalizing_duration(df)
    txl_vie, vie_txl, vie_osl, osl_vie = define_routes(df)
    temp_df, temp2_df = delay_plots(df)
    statistics(df)



