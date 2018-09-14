import pandas as pd
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    # pandas settings information
    pd.set_option('display.max_rows', 20)
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

    # display the column names of the DataFrame
    # print(list(data.columns))

    '''
    Checking data quality 
    '''
    # get some basic statistics about the data set
    # print(data.describe())
    # minimum average temperature is -26 can this be correct ==> Harbin, China 1922-01-01
    # print(data.loc[data['AverageTemperature'] == -26.772, :])

    # display the count of rows where the specified column is missing
    for i in list(data.columns):
        missing_values = unknown_rows_per_col(i)
        number_of_rows = missing_values.shape[0]
        # print('Number of missing values in column {}: {}'.format(i, number_of_rows))

    # conclusion: there are only missing data for the columns AverageTemperature and AverageTemperatureUncertainty

    # save the unknown rows where the value in the row AverageTemperature is Null (missing)
    unknown_values = unknown_rows_per_col('AverageTemperature')

    # max unknown date in the data set (the dt.date is used to convert the datetime to a date only)
    # max_unknown_date = unknown_values['dt'].dt.date.max()
    # find the rows which have NaN in the AverageTemperature for the max missing date
    # print(unknown_values.loc[unknown_values['dt'].dt.date == max_unknown_date, :])

    # display the number of missing values per group
    unknown_values1 = unknown_values.drop(['dt', 'AverageTemperatureUncertainty', 'Latitude', 'Longitude'], axis=1)
    # fill the missing values with 0's ==> counting missing values per group in pandas is difficult
    # rename AverageTemperature to number_of_missing_values per country and city for the column AverageTemperature
    unknown_values2 = unknown_values1.fillna(0).groupby(['Country', 'City']).count().rename(columns={'AverageTemperature':'number_of_missing_values'})
    # print(unknown_values2)
    # validate the number of rows missing
    # print(unknown_values.loc[unknown_values['City'] == 'Melbourne', :])

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

    # define city, country and dt as index
    data = data.set_index(['dt'])

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
    # print(data)

    '''
    check the data quality again
    '''

    for i in list(data.columns):
        missing_values = unknown_rows_per_col(i)
        number_of_rows = missing_values.shape[0]
        print('Number of missing values in column {}: {}'.format(i, number_of_rows))

    # conclusion: there aren't missing data anymore ==> data is clean and ready for analysis

    # TODO discuss: observations missing for some countries
    # TODO ==> not all countries have measures of the average temperature since 1743

    '''
    Data analysis
    '''
    # print(data.describe())

    # the number of occurrences per year, country and city
    un_y_c_c = data.groupby(['year', 'Country', 'City']).nunique()
    # print(un_y_c_c)

    agg_d = data.loc[data['City'] == 'Moscow'].groupby(['year']).mean()

    agg_d.plot.line(y='AverageTemperature')
    plt.show()

    # TODO: Discuss what are we going to do with the missing values?
    # TODO: Take the average of the same city of x years?
    # TODO: Leaving them out will have a significant impact on the results.

    print('\nIt works')

