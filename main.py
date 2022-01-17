# Case 5 BIS - Lucas Winkler
# import libaries
import pandas as pd
import numpy as np
from scipy.stats import shapiro
import os
from matplotlib import pyplot as plt
# necessary for the robust parameter of the regression plot, do not comment/delete!
import statsmodels
import seaborn as sns
from scipy.stats import mannwhitneyu

# import of raw data + seperated df without cancelled flights for calculations
def raw_data():
    path = 'C:/Users/Lucas/Desktop/Master/01_Semester/Betriebliche Informationssysteme/Case 5 - FlyUIBK/raw_data.csv'
    try:
        normPath = path.replace(os.sep, '/')
    except:
        print('Error while normalizing path.')
    df = pd.read_csv(normPath, sep=';')
    df_compl = pd.read_csv(normPath, sep=';')
    df.fillna(0)
    df_compl.fillna(0)
    indexNames = df[df['Actual arrival time'] == 'Cancelled'].index
    df.drop(indexNames, inplace=True)
    df = df.astype({"Arrival delay in minutes": 'int64', "Day of Week": 'int64', "Delay indicator": 'int64',
                    "normalized delay indicator": 'int64'})
    print(df.dtypes)
    print(df_compl.dtypes)
    return df, df_compl

# plot delays as a bar plot
def delay_plots(df):
    # print(df.dtypes)
    # plot occurences of delays per day of week for FlyUIBK
    temp_df = df.loc[df['Delay indicator'] == 1]
    temp_df = temp_df.loc[df['Airline'] == 'FlyUIBK']
    temp_df = (temp_df.groupby(['Arrival delay in minutes', 'Day of Week']).size())
    # temp_df.groupby('Day of Week').count().plot(x='Day of Week', y='Arrival delay in minutes', kind='bar')
    # plt.show()
    # plot occurences of delays per day of week for FlyUIBK
    temp2_df = df.loc[df['Delay indicator'] == 1]
    temp2_df = temp2_df.loc[temp2_df['Airline'] == 'LDA']
    temp2_df = (temp2_df.groupby(['Arrival delay in minutes', 'Day of Week']).size())
    # temp2_df.groupby('Day of Week').count().plot(x='Day of Week', y='Arrival delay in minutes', kind='bar')
    # plt.show()
    return temp_df, temp2_df

# several statistical methods, a little messy and might be better to split up on several functions but works....
# commented some unneeded features of the function to reduce computation time. Feel free to uncomment and test
def statistics(df):
    # plot distribution of delays per airline
    temp_df = df.loc[df['Airline'] == 'FlyUIBK']
    # print(temp_df['Arrival delay in minutes'].describe())
    temp_df['Arrival delay in minutes'].value_counts().sort_index().plot(kind='bar')
    plt.axvline(x=22, color='#DC143C')
    plt.show()
    temp2_df = df.loc[df['Airline'] == 'LDA']
    flyuibk_perc_delayed = (temp_df['Delay indicator'][(temp_df['Delay indicator'] == 1)].count()) / (
        (temp_df['Delay indicator']).count())
    lda_perc_delayed = (temp2_df['Delay indicator'][(temp2_df['Delay indicator'] == 1)].count()) / (
        (temp2_df['Delay indicator']).count())
    print((temp2_df['normalized delay indicator'][(temp2_df['normalized delay indicator'] == 1)].count()))
    lda_norm_perc_delayed = (temp2_df['normalized delay indicator'][
                                 (temp2_df['normalized delay indicator'] == 1)].count()) / (
                                (temp2_df['normalized delay indicator']).count())
    print('FlyUIBK percentage of delayed flights: ', round(flyuibk_perc_delayed * 100, 2))
    print('LDA percentage of delayed flights: ', round(lda_perc_delayed * 100, 2))
    print('LDA normalized percentage of delayed flights: ', round(lda_norm_perc_delayed * 100, 2))

    # print(temp2_df['Arrival delay in minutes'].describe())
    temp2_df['Arrival delay in minutes'].value_counts().sort_index().plot(kind='bar')
    plt.axvline(x=19, color='#DC143C')
    plt.show()

    temp2_df['normalized_delay'].value_counts().sort_index().plot(kind='bar')
    plt.axvline(x=14, color='#DC143C')
    plt.show()
    sns.scatterplot(data=temp_df, y='Arrival delay in minutes', x='Number of passengers', hue='route')
    sns.regplot(data=temp_df, y='Arrival delay in minutes', x='Number of passengers', scatter=False)
    plt.show()
    sns.scatterplot(data=temp_df, y='Arrival delay in minutes', x='Number of passengers', hue='route')
    sns.regplot(data=temp_df, y='Arrival delay in minutes', x='Number of passengers', scatter=False, robust=True)
    plt.show()
    #corr, p_value = scipy.stats.pearsonr(scatter_x,scatter_y)
    temp3_df = df.loc[df['Airline'] == 'FlyUIBK']
    sns.stripplot(data=temp3_df, y='Arrival delay in minutes', x='day', hue='day',
                  order=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], palette='deep')
    plt.axhline(y=15, color='#DC143C')
    plt.show()
    temp4_df = df.loc[df['Airline'] == 'LDA']
    sns.stripplot(data=temp4_df, y='Arrival delay in minutes', x='day', hue='day',
                  order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],palette='deep')
    plt.axhline(y=15, color='#DC143C')
    plt.show()
    sns.countplot(x='day', data=temp_df,order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],palette='deep')
    plt.show()
    sns.countplot(x='day', data=temp2_df,order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],palette='deep')
    plt.show()


    # some statistical metrics, in the end not needed in the report due to low information gain
    print('Mean delay of FlyUIBK in minutes:', round(temp_df['Arrival delay in minutes'].mean(), 2))
    print('Mean delay of LDA in minutes:', round(temp2_df['Arrival delay in minutes'].mean(), 2))
    print('Median delay of FlyUIBK in minutes:', round(temp_df['Arrival delay in minutes'].median(), 2))
    print('Median delay of LDA in minutes:', round(temp2_df['Arrival delay in minutes'].median(), 2))
    print('Mean delay of LDA (norm) in minutes:', round(temp2_df['normalized_delay'].mean(), 2))
    print('Median delay of LDA (norm) in minutes:', round(temp2_df['normalized_delay'].median(), 2))
    return

# hypothesis test to find out whether LDA truly performs better than FlyUBIK regarding the delay times
# shapiro wilk test + mann whitney U / wilcoxon rank sum chosen as test methods
# shapiro wilk because of power @Rivalis referenced in report
# mann whitney test/wilcoxon rank sum because of non-parametric data
def hypothesis(df):
    temp7_df = df.loc[(df['Airline'] == 'FlyUIBK') & (df['Arrival delay in minutes'] > 0)].copy()
    temp8_df = df.loc[(df['Airline'] == 'LDA') & (df['Arrival delay in minutes'] > 0)].copy()
    temp7_df['lognorm'] = np.log(temp7_df['Arrival delay in minutes'])
    temp8_df['lognorm'] = np.log(temp8_df['Arrival delay in minutes'])
    stat, p = mannwhitneyu(temp7_df['Arrival delay in minutes'], temp8_df['Arrival delay in minutes'], alternative='two-sided')
    stat_less, p_less = mannwhitneyu(temp7_df['Arrival delay in minutes'], temp8_df['Arrival delay in minutes'], alternative='less')
    print(stat, p)
    print(stat_less, p_less)
    return

def shapiro_wilk(df):

    temp5_df = df.loc[(df['Airline'] == 'FlyUIBK') & (df['Arrival delay in minutes'] > 0)].copy()
    temp6_df = df.loc[(df['Airline'] == 'LDA') & (df['Arrival delay in minutes'] > 0)].copy()
    temp5_df['lognorm'] = np.log(temp5_df['Arrival delay in minutes'])
    temp6_df['lognorm'] = np.log(temp6_df['Arrival delay in minutes'])
    shapiro_flyuibk = shapiro(temp5_df['lognorm'])
    shapiro_lda = shapiro(temp6_df['lognorm'])
    shapiro_lda_norm = shapiro(temp6_df['lognorm'])
    print(shapiro_flyuibk)
    print(shapiro_lda)
    print(shapiro_lda_norm)
    print('flyuibk: {:.10f}'.format(shapiro_flyuibk.statistic))
    print('lda: {:.10f}'.format(shapiro_lda.statistic))
    print('lda adj: {:.10f}'.format(shapiro_lda_norm.statistic))
    print('Shapiro FlyUIBK: {:.18f}'.format(shapiro_flyuibk.pvalue))
    print('Shapiro LDA: {:.18f}'.format(shapiro_lda.pvalue))
    print('Shapiro LDA normalized: {:.18f}'.format(shapiro_lda_norm.pvalue))
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df, df_compl = raw_data()
    # uncomment for other features
    temp_df, temp2_df = delay_plots(df)
    statistics(df)
    shapiro_wilk(df)
    hypothesis(df)
