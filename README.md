# Master in Big Data Solutions Team 1

Welcome to the bts-mbds-precourse-team1 repository. In this README you will find all the information needed to start using this project to analyse the 'Climate Change: Earth Surface Temperature Data'. 

**Last Update: September 28, 2018**

# Names and contact information

| Name | Mail |
| ---- | ---- |
| Chico | francisco.coreas@bts.tech  |
| Alan  | alan.kwan@bts.tech  |
| Jonas  | jonas.cristens@bts.tech  |

# Project Description / Abstract

This project examines a collection of global temperature time series for major cities collected by Berkeley Earth. The series consists of monthly data starting from 1750 until 2013. We developed Python code to parse, clean, manipulate, and analyze the data, allowing us to gain insights about how much temperatures have changed over time, how the rate of warming has changed, and how the warming varies geographically. Finally, we developed an ARIMA (autoregressive integrated moving average) model based on the data to project future temperature changes.

The analysis shows significant warming overall, particularly starting around 1920.  Although all regions studied experienced warming, the amount was not uniform geographically.  The ARIMA model projects the warming trend to continue in the future.

# Dataset

For this project we have used following data: 'Climate Change: Earth Surface Temperature Data' from [kaggle](https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data#GlobalTemperatures.csv). The website offers different data sets, the data set used in this project is: GlobalLandTemperaturesByMajorCity.csv which is available via the link below.  
Data source: https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data/home

The data set contains following 7 columns:
* dt: date of the recording in the following format YYYY-MM-DD
* AverageTemperature: the average temperature in Celsius
* AverageTemperatureUncertainty: the uncertainty in the the previous measure (AverageTemperature)
* City: the city where the average temperature was measured
* Country: the country where the average temperature was measured
* Latitude: the latitude where the temperature was measured
* Longitude: the longitude where the temperature was measured

# Files and usage
The files and folders available in this project are the following:
* main.py
* /figures
  * contains graphs made during this project e.g. the average temperature between 1850 and 2012
* /data_set
  * you have to put the climate change data set here.  ([click here to download the data set](https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data#GlobalTemperatures.csv))  main.py will look in this folder for the data set when it runs 
  * Note: put following CSV in this folder (from the link above) GlobalLandTemperaturesByMajorCity.csv

For example

    main.py: This file contains the core of the project (data cleaning, data exploration and data visualization)
    
**Usage**: Run the following command line into your terminal:

    python main.py
    
**Note:** don't forget to put the data set (GlobalLandTemperaturesByMajorCity.csv) in following folder:
* /data_set

You can dowload the data set via the following link: [https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data#GlobalTemperatures.csv](https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data#GlobalTemperatures.csv) use the following csv: GlobalLandTemperaturesByMajorCity

# Examples

**Average temperatue prediction:**
![alt text](https://github.com/bts-mbds-precourse/bts-mbds-precourse-team1/blob/master/figures/average_temperature_prediction.png "temperature prediction untill 2050")

**Average temperatue (rolling 10 year average) from 1850 untill 2012:** 
![alt text](https://github.com/bts-mbds-precourse/bts-mbds-precourse-team1/blob/master/figures/pol_reg_rolling_mean.png "average temperature over time")

**Increase in temperature per city between 1850 and 2012:** 
![alt text](https://github.com/bts-mbds-precourse/bts-mbds-precourse-team1/blob/master/figures/increase_1850_2012.png "average temperature increase per city between 1850 and 2012")

**Increase in temperature between 1850 and 2012 in bins of 10 for latitude and longitude:** 
![alt text](https://github.com/bts-mbds-precourse/bts-mbds-precourse-team1/blob/master/figures/increase_1850_2012_lat_lon_bin.png "Increase in temperature in bins of 10 for latitude and longitude")

**For more graphs go to:**
[link to the plots](https://github.com/bts-mbds-precourse/bts-mbds-precourse-team1/tree/master/figures)
    
# References
Data source: https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data/home
Released Under following license: CC BY-NC-SA 4.0 
