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
    path = '/Users/jonascristens/Documents/BTS/precourse/project/GlobalLandTemperaturesByMajorCity.csv'
    # reading the CSV file containing the climate change data
    data = pd.read_csv(path)


    # convert the dt column to a pandas datetime64 type
    data.loc[:, 'dt'] = pd.to_datetime(data.loc[:, 'dt'], infer_datetime_format=True)

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

    # fill the missing values for the column AverageTemperature with the mean per country, city and month
    data['AverageTemperature'] = data\
        .groupby(['City', 'Country', 'month'])['AverageTemperature']\
        .transform(lambda x: x.fillna(x.mean()))
    # fill the missing values for the column AverageTemperatureUncertainty with the mean per country, city and month
    data['AverageTemperatureUncertainty'] = data\
        .groupby(['City', 'Country', 'month'])['AverageTemperatureUncertainty']\
        .transform(lambda x: x.fillna(x.mean()))

    # drop the month column
    data.drop('month', inplace=True, axis=1)

    '''
    check the data quality again
    '''
    # display the count of rows where the specified column is missing
    missing_stats()

    # conclusion: there aren't missing data ==> data is clean and ready for analysis

    # TODO discuss: observations missing for some countries
    # TODO ==> not all countries have measures of the average temperature since 1743

    '''
    Data restructuring and analysis
    '''
    # get some basic statistics about the data set
    # print(data.describe())

    # the number of occurrences per year
    # un_y_c_c = data.groupby(['year']).nunique()
    # del un_y_c_c

    # retrieve the cities which have observations in the year 1744
    agg_y_c = data.groupby(['year', 'City']).mean()
    cities_in_year = agg_y_c.loc[1850].index.values
    print(len(cities_in_year))
    # print(cities_in_year)
    del agg_y_c

    # retrieve the count of observations for the cities which have observations since 1744
    # purpose: validate that no city is missing ==> could influence the result
    '''number_of_obs_per_year = data.loc[data['City'].isin(cities_in_year)]\
        .groupby(['year'])\
        .count()'''

    # select only data from cities which have observations since 1744 and group by year ==> aggregation average
    agg_d = data.loc[data['City'].isin(cities_in_year)]\
        .groupby(['year'])\
        .mean()
    del cities_in_year

    # take the rolling average over a 10 year period where year greater than 1744
    agg_rolling = agg_d.loc[(agg_d.index >= 1850) & (agg_d.index < 2013)].rolling(10, min_periods=1).mean()

    del agg_d
    # del number_of_obs_per_year

    # print(agg_rolling)

    agg_rolling.plot.line(y='AverageTemperature')
    plt.show()

    # TODO: Discuss what are we going to do with the missing values?
    # TODO: Take the average of the same city of x years?
    # TODO: Leaving them out will have a significant impact on the results.
    # options shrink the data frame
    # anomaly temperature = current average - avg(over the entire time period) or some other specified time period
    # missing data, replace it with in different ways interpolation
    # regression ==> over the slope calculate the significance (in comparison with 0)
    # bar chart which represent the increase in temperature

    print(stats.ttest_rel(agg_rolling.loc[1850], agg_rolling.loc[2012]))

    agg_rolling.reset_index(level=0, inplace=True)

    sns.lmplot(x='year', y="AverageTemperature", data=agg_rolling.loc[agg_rolling['year'] > 1850, :], order=3)
    plt.show()
    print('\nIt works')

