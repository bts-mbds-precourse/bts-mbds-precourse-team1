import pandas as pd

# pandas settings data
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# local path to the file containing the temperature data
path = '/Users/jonascristens/Documents/BTS/precourse/project/GlobalLandTemperaturesByMajorCity.csv'

# reading the CSV file containing the climate change data
data = pd.read_csv(path)
print(data)

