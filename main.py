import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


def unknown_rows_per_col(col_name):
    """
    Parameters:
        :param col_name: col_name: the name of the column on which you want to perform following action

    Return:
        Return all rows where col_name(column name) has empty values
    """
    return data.loc[data[col_name].isnull(), :]


def count_per_column(col_name):
    """
    Parameters:
         :param col_name: (String): col_name: the name of the column on which you want to perform following action

    Returns:
         Return the count per (grouped by) col_name e.g. return the count of rows per City
     """
    return data.loc[:, col_name].value_counts()


def missing_stats():
    """
    Returns:
         Return stats about the missing value in the data set
    """
    for i in list(data.columns):
        missing_values = unknown_rows_per_col(i)
        number_of_rows = missing_values.shape[0]
        print('Number of missing values in column {}: {}'.format(i, number_of_rows))


if __name__ == '__main__':

    # pandas settings information
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    '''
    General data importing
    '''
    # local path to the file containing the temperature data
    path = './data_set/GlobalLandTemperaturesByMajorCity.csv'
    # reading the CSV file containing the climate change data
    data = pd.read_csv(path)
#    #create data2 array with same data; we will fill missing values with average of month
#    data2 = pd.read_csv(path)

    # convert the dt column to a pandas datetime64 type
    data.loc[:, 'dt'] = pd.to_datetime(data.loc[:, 'dt'], infer_datetime_format=True)
#    data2.loc[:, 'dt'] = pd.to_datetime(data2.loc[:, 'dt'], infer_datetime_format=True)

    '''
    Checking data quality 
    '''
    # get some basic statistics about the data set
    # print(data.describe())
    # minimum average temperature is -26 can this be correct ==> Harbin, China 1922-01-01
    # print(data.loc[data['AverageTemperature'] == -26.772, :])

    # display the count of rows where the specified column is missing
    # missing_stats()

    # conclusion: there are only missing data for the columns AverageTemperature and AverageTemperatureUncertainty

    # save the unknown rows where the value in the row AverageTemperature is Null (missing)
    # unknown_values = unknown_rows_per_col('AverageTemperature')

    # max unknown date in the data set (the dt.date is used to convert the datetime to a date only)
    # max_unknown_date = unknown_values['dt'].dt.date.max()
    # find the rows which have NaN in the AverageTemperature for the max missing date
    # print(unknown_values.loc[unknown_values['dt'].dt.date == max_unknown_date, :])

    # display the number of missing values per group
    # unknown_values1 = unknown_values.drop(['dt', 'AverageTemperatureUncertainty', 'Latitude', 'Longitude'], axis=1)
    # fill the missing values with 0's ==> counting missing values per group in pandas is difficult
    # rename AverageTemperature to number_of_missing_values per country and city for the column AverageTemperature
    '''unknown_values2 = unknown_values1.fillna(0)\
        .groupby(['Country', 'City'])\
        .count()\
        .rename(columns={'AverageTemperature':'number_of_missing_values'})
    print(unknown_values2)
    # validate the number of rows missing
    # print(unknown_values.loc[unknown_values['City'] == 'Melbourne', :])'''

    '''
    Data manipulation to increase the data quality of the data set
    '''
    # grouped_by_city = data.groupby(['City', 'Country']).mean()

    # length of unique countries
    # print(len(data.Country.unique()))
    # count of rows per city
    # print(count_per_column('City'))

    # extract the month from the dt column to group later
    data['month'] = data['dt'].dt.month
    data['year'] = data['dt'].dt.year
 #   data2['month'] = data2['dt'].dt.month
 #   data2['year'] = data2['dt'].dt.year

#look for NA's per year
    na_temps=data.loc[data['AverageTemperature'].isna()] #get dataframe with only NaN in
                                                         # AverageTemperature
#    print(na_temps.shape)
#    print(na_temps)
    # us df.value_counts() to get two columns, years and number of NaNs for that year.  Use
    # .keys().tolist() and .tolist() to save these columns as two lists that can be
    # plotted against each other
    na_cts=na_temps['year'].value_counts(sort=False) #if sort = True, will not give years
                                                     # in chronological order
                                                     #but instead by order of frequency
    year_na=na_temps['year'].value_counts(sort=False).keys().tolist()
    cts_na=na_temps['year'].value_counts(sort=False).tolist()

#the above lists do not include years with zero NaN;
#  do for loop to add these years in years list and zeros in counts list
    for i in range(1744, 2013+1):
        if i not in year_na:
            year_na.append(i)
            cts_na.append(0)

#now, for plotting, remove years out of range

    plot_year_na=[]
    plot_cts_na=[]
    for i,v in enumerate(year_na):
        if v>=1850 and v<=1900:
            plot_year_na.append(v)
            plot_cts_na.append(cts_na[i])

    plt.plot(plot_year_na, plot_cts_na, 'o')
    plt.xlabel('Year')
    plt.ylabel('Missing points')
    plt.suptitle('Number of Missing Data Points per Year')
    plt.show()
#    print(data.loc['year'==1868])
#    plt.plot(na_cts[:,1], x='year', y)

#    print(data.loc[(data['year'] >= 1865) & (data['year']<=1866), :])
#    print(data.loc[data['year'].isin([1865, 1866])])

#    print(data.loc[data['AverageTemperature'] == -26.772, :])

    #     # retrieve the cities which have observations in the year 1744
    agg_y_c = data.groupby(['year', 'City']).mean()
    cities_in_year = agg_y_c.loc[1850].index.values
    print(cities_in_year)
#    missing_data=data.loc[(data['City'].isin(cities_in_year)) & (data['year'].isin([1865, 1866, 1867]))]


    # fill the missing values for the column AverageTemperature with interpolated data
    # from the same city
    data['InterpolatedTemperature'] = data\
        .groupby(['City'])['AverageTemperature']\
        .transform(lambda x: x.fillna(x.interpolate()))
    #fill in missing data with average of temperatures from that month
    data['MeanFilledTemperature'] = data\
        .groupby(['City', 'month'])['AverageTemperature']\
        .transform(lambda x: x.fillna(x.mean()))
    #fill in missing data by interpolating between closest data points of same month
    data['MeanFilledTemperature2'] = data\
        .groupby(['City', 'month'])['AverageTemperature']\
        .transform(lambda x: x.fillna(x.interpolate()))


    # fill the missing values for the column AverageTemperatureUncertainty with the mean per country, city and month
    data['AverageTemperatureUncertainty'] = data\
        .groupby(['City', 'Country', 'month'])['AverageTemperatureUncertainty']\
        .transform(lambda x: x.fillna(x.mean()))

# now look at data frame with only nan values, with mean and interpolated temps,
# focusing on years of greatest discrepancy
    data_nan=data[(data['AverageTemperature'].isna()) & (data['year']>1865) & (data['year']<1870)]
##    print(data_nan)
##    data_nan.plot(y=['InterpolatedTemperature', 'MeanFilledTemperature'], style ='.')
##    print(data_nan['InterpolatedTemperature'].nunique())
##    print(data_nan['MeanFilledTemperature'].nunique())
##    plt.show()
##    plt.bar(['Interpolated', 'Mean-filled'], [data_nan['InterpolatedTemperature'].nunique(), data_nan['MeanFilledTemperature'].nunique()])
##    plt.suptitle('Number of unique values replacing missing data')
##    plt.show()
    plt.scatter(data_nan['MeanFilledTemperature'], data_nan['InterpolatedTemperature'])
    plt.show()

#Focus on areas of coldest interpolated data and hottest mean-filled data
    data_nan2=data[(data['AverageTemperature'].isna()) & (data['year']>1865) & (data['year']<1870) & (data['InterpolatedTemperature']<15) & (data['MeanFilledTemperature']>25)]
    print(data_nan2['City'].unique())

#from looking at data_nan2, areas where interpolated temp is much bigger than mean temp
#are in India

    india=data[((data['City'] == 'Delhi') | (data['City'] == 'Faisalabad') | (data['City'] == 'Lahore') | (data['City'] == 'New Delhi')) & (data['year']>1855) & (data['year']<1875)]
#    india.plot(y=['AverageTemperature', 'InterpolatedTemperature', 'MeanFilledTemperature'], style='.')
    delhi=india[india['City']=='Delhi']
    print(delhi)
    plt.plot(delhi['dt'], delhi['AverageTemperature'], 'o')
    plt.plot(delhi['dt'], delhi['InterpolatedTemperature'], 'x')
    plt.plot(delhi['dt'], delhi['MeanFilledTemperature'], '+:')
    plt.plot(delhi['dt'], delhi['MeanFilledTemperature2'], '1:')
    plt.show()
#    print(india)
#    interp_data=data.loc[(data['City'].isin(cities_in_year)) & (data['year'].isin([1865, 1866, 1867]))]
#    mean_data=data2.loc[(data2['City'].isin(cities_in_year)) & (data2['year'].isin([1865, 1866, 1867]))]

#    print([interp_data['City']=='Abidjan'])

#    print(data.loc[data['AverageTemperature'] == -26.772, :])

#     # drop the month column
#     data.drop('month', inplace=True, axis=1)
#
#     '''
#     check the data quality again
#     '''
#     # display the count of rows where the specified column is missing
# #    missing_stats()
#
#     # conclusion: there aren't missing data ==> data is clean and ready for analysis
#
#     # TODO discuss: observations missing for some countries
#     # TODO ==> not all countries have measures of the average temperature since 1743
#
#     '''
#     Data restructuring and analysis
#     '''
#     # get some basic statistics about the data set
#     # print(data.describe())
#
#     # the number of occurrences per year
#     # un_y_c_c = data.groupby(['year']).nunique()
#     # del un_y_c_c
#
#    #     # retrieve the cities which have observations in the year 1744
#    agg_y_c = data.groupby(['year', 'City']).mean()
#    cities_in_year = agg_y_c.loc[1850].index.values
#    print(cities_in_year)
#    print(data.loc[(data['year'] >= 1865) & (data['year']<=1866), :])

#    #Testing print statements for pandas DataFrames
#    print(data.loc[data['City'].isin(cities_in_year)])
#    print(data.loc[(data['City'].isin(cities_in_year)) & (data['year'].isin([1865, 1866]))])
#    print(data.loc[(data['City'].isin(cities_in_year)) & (data['year'].isin([1865, 1866])), :])

#    print(data.loc[data['year'].isin([1865, 1866])])

#    print(data.loc[(data['year'].isin([1865, 1866])) and (data['City'].isin(cities_in_year)), :])


#    print(data.loc[(data['year'] == 1865) and (data['year'] == 1866) and (data['City'].isin(cities_in_year)) :])

#     # print(len(cities_in_year))
#     # print(cities_in_year)
#     del agg_y_c
#
     # retrieve the count of observations for the cities which have observations since 1744
     # purpose: validate that no city is missing ==> could influence the result
#     '''number_of_obs_per_year = data.loc[data['City'].isin(cities_in_year)]\
#         .groupby(['year'])\
#         .count()'''

     # select only data from cities which have observations since 1744 and group by year ==> aggregation average
    agg_d = data.loc[data['City'].isin(cities_in_year)]\
        .groupby(['year'])\
        .mean()
    #for data with means replacing NaN
#    agg_d2 = data2.loc[data2['City'].isin(cities_in_year)]\
#        .groupby(['year'])\
#        .mean()

#     del cities_in_year
#
#     # take the rolling average over a 10 year period where year greater than 1744
    agg_rolling = agg_d.loc[(agg_d.index >= 1850) & (agg_d.index < 1900)].rolling(10, min_periods=1).mean()
 #   agg_rolling2 = agg_d2.loc[(agg_d2.index >= 1850) & (agg_d2.index < 2013)].rolling(10, min_periods=1).mean()
#    print(agg_rolling)
#    agg_rolling['temp2']=agg_rolling2['AverageTemperature']
#    print(agg_rolling)
    #     print(agg_rolling.loc[1860:1870])
#     del agg_d
#     # del number_of_obs_per_year
#
#     # print(agg_rolling)
#  #   print(data.loc[agg_rolling=min(agg_rolling)])
#    plt.plot(y=agg_rolling['AverageTemperature'])#,'b', x=agg_rolling2['year'], y=agg_rolling2['AverageTemperature'],'r' )
#    plt.plot(y=agg_rolling2['AverageTemperature'])
#    agg_rolling.plot.line(y=['InterpolatedTemperature', 'MeanFilledTemperature'])
#    plt.show()

#    agg_rolling2.plot.line(y='AverageTemperature')
#    plt.show()
#
    agg_rolling.plot(y=['InterpolatedTemperature', 'MeanFilledTemperature', 'MeanFilledTemperature2'], style='o')
    plt.suptitle('Annual Mean Temperature for Different Missing Data Strategies')
    plt.show()

#     # TODO: Discuss what are we going to do with the missing values?
#     # TODO: Take the average of the same city of x years?
#     # TODO: Leaving them out will have a significant impact on the results.
#     # options shrink the data frame
#     # anomaly temperature = current average - avg(over the entire time period) or some other specified time period
#     # missing data, replace it with in different ways interpolation
#     # regression ==> over the slope calculate the significance (in comparison with 0)
#     # bar chart which represent the increase in temperature
#
#     # print(stats.ttest_rel(agg_rolling.loc[1850], agg_rolling.loc[2012]))
#
#     agg_rolling.reset_index(level=0, inplace=True)
#
#    sns.lmplot(x='year', y="AverageTemperature", data=agg_rolling.loc[agg_rolling['year'] > 1850, :], order=3)
#     plt.show()
#     print('\nIt works')
#
