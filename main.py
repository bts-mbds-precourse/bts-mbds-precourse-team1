import pandas as pd


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

    # local path to the file containing the temperature data
    path = '/Users/jonascristens/Documents/BTS/precourse/project/GlobalLandTemperaturesByMajorCity.csv'

    # reading the CSV file containing the climate change data
    data = pd.read_csv(path)

    # convert the dt column to a pandas datetime64 type
    data.loc[:, 'dt'] = pd.to_datetime(data.loc[:, 'dt'], infer_datetime_format=True)

    # display the column names of the DataFrame
    print(list(data.columns))

    # display the rows where the specified column is unknown
    for i in list(data.columns):
        #print(i)
        missing_values = unknown_rows_per_col(i)
        number_of_rows = missing_values.shape[0]
        #print(unknown_rows_per_col(i))
        print('Number of missing values in column {}: {}'.format(i, number_of_rows))

    # conclusion: there is only missing data for the columns AverageTemperature and AverageTemperatureUncertainty

    unknown_values = unknown_rows_per_col('AverageTemperature')

    # max unknown date in the data set (the dt.date is used to convert the datetime to a date only)
    max_unknown_date = unknown_values['dt'].dt.date.max()

    # find the rows which have NaN in the AverageTemperature for the max missing date
    #print(unknown_values.loc[unknown_values['dt'].dt.date == max_unknown_date, :])

    # count of rows per city
    #print(count_per_column('City'))

    # TODO: Discuss what are we going to do with the missing values?
    # TODO: Take the average of the same city of x years?
    # TODO: Leaving them out will have a significant impact on the results.

    print('It works')

