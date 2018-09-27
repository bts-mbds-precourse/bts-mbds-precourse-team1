# Master in Big Data Solutions Team 1

Welcome to the bts-mbds-precourse-team1 repository. In this README you will find all the information needed to start using this project to analyse the 'Climate Change: Earth Surface Temperature Data'. In the section examples are some examples of things we found in the data set.

**The repository structure is still taking shape, be aware that the structure can change a lot in the coming days.**

# Names and contact information

| Name | Mail |
| ---- | ---- |
| Chico | francisco.coreas@bts.tech  |
| Alan  | alan.kwan@bts.tech  |
| Jonas  | jonas.cristens@bts.tech  |

# Project Description / Abstract

This project examines a collection of global temperature time series for major cities collected by Berkeley Earth from other primary sources.  The series consists of monthly data starting from 1750 until 2017.  We develop Python code to download the data and convert it into data frames, a format which allows us to write Python code to clean, manipulate, and analyze the data. We further to look for insights from the data, such as the extent of temperature change in major cities and possible factors driving these differences.  

# Dataset

For this project we have used following data: 'Climate Change: Earth Surface Temperature Data' from https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data/home. The website offers different data sets, the data set used in this project is: GlobalLandTemperaturesByMajorCity.csv which is available via the link below. 
Data source: https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data/home

The data set contains the following 7 columns:
* dt: date of the recording in the following format YYYY-MM-DD
* AverageTemperature: the average temperature in Celsius
* AverageTemperatureUncertainty: the uncertainty in the the previous measure (AverageTemperature)
* City: the city where the average temperature was measured
* Country: the country where the average temperature was measured
* Latitude: the latitude where the temperature was measured
* Longitude: the longitude where the temperature was measured

# Files and usage
The files available in this project are the following:
* main.py

For example

    main.py: This file contains the core of the project (data cleaning, data exploration and data visualization)
    
**Usage**: Run the following command line into your terminal:

    python main.py
    
**Note:** don't forget to put the dataset (GlobalLandTemperaturesByMajorCity.csv) in the following folder:
* /data_set

You can dowload the dataset via the following link: https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data#GlobalTemperatures.csv

# Examples

**Average temperatue prediction: **
![alt text](https://github.com/bts-mbds-precourse/bts-mbds-precourse-team1/blob/master/figures/average_temperature_prediction.png "temperature prediction")

**Average temperatue (rolling 10 year average) from 1850 untill 2012:** 
![alt text](https://github.com/bts-mbds-precourse/bts-mbds-precourse-team1/blob/master/figures/pol_reg_rolling_mean.png "average temperature")

**Increase in temperature between per city 1850 untill 2012:** 
![alt text](https://github.com/bts-mbds-precourse/bts-mbds-precourse-team1/blob/master/figures/increase_1850_2012.png "average temperature increase per city between 1850 and 2012")

**Increase in temperature in bins of 10 for latitude and longitude:** 
![alt text](https://github.com/bts-mbds-precourse/bts-mbds-precourse-team1/blob/master/figures/increase_1850_2012_lat_lon_bin.png "Increase in temperature in bins of 10 for latitude and longitude")

**For more result go to:**
[links to the plots](https://github.com/bts-mbds-precourse/bts-mbds-precourse-team1/tree/master/figures)
    
# References
Data source: https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data/home
Released Under following license: CC BY-NC-SA 4.0 
