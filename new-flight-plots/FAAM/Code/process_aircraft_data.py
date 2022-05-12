import os
import numpy as np
import pandas as pd
from   netCDF4 import Dataset
from   datetime import datetime

"""
Code to read the observational data from the FAAM aircraft NetCDF file and extract
the variables of interest into a CSV file.
"""

# Define the path to the original NetCDF file containing all data and the path to
# the new CSV file you wish to write the data to.
aircraft_nc_file  = '../Data_Files/Aircraft/core_faam_20180629_v004_r0_c110_1hz_with_PM25.nc'
aircraft_csv_file = '../Data_Files/Aircraft/C110_20180629_Aircraft_Track_Data.csv'

# Read the original data.
f = Dataset(aircraft_nc_file,mode='r',format='NETCDF4')

# Create a dataframe to hold the extracted data.
df = pd.DataFrame()

# Read the time data and convert from units of 'seconds from 2018-06-29 07:12:05' to datetime
# and seconds since 1970.
time = f.variables['Time'][:]
epoch = datetime(1970,1,1)
start_time = (datetime(2018,6,29,7,12,5) - epoch).total_seconds()
flight_times = [x+start_time for x in time]
print(flight_times[0])
datetime_data = [pd.to_datetime(x,unit='s') for x in flight_times]
print(datetime_data[0])
df['Time / seconds since 1970-01-01 00:00:00 UTC'] = pd.Series(flight_times,index=datetime_data)

# Read the latitude data.
lat = f.variables['LAT_GIN'][:]
df['Latitude / degrees north'] = pd.Series(lat,index=datetime_data)

# Read the longitude data.
lon = f.variables['LON_GIN'][:]
df['Longitude / degrees east'] = pd.Series(lon,index=datetime_data)

# Read the altitude data.
alt = f.variables['ALT_GIN'][:]
df['Altitude / m'] = pd.Series(alt,index=datetime_data)

# Read the ozone data.
o3 = f.variables['O3_TECO'][:]
df['O3 / ppb'] = pd.Series(o3,index=datetime_data)

# Read the carbon monoxide data.
co = f.variables['CO_AERO'][:]
df['CO / ppb'] = pd.Series(co,index=datetime_data)

# Read the particulate matter data.
pm = f.variables['PM2.5'][:]
df['PM2.5 / ug m-3'] = pd.Series(pm,index=datetime_data)

# Filter to remove data which have no location information.
#df = df[np.isfinite(df['Latitude / degrees north'])]
#df = df[np.isfinite(df['Longitude / degrees east'])]
#df = df[np.isfinite(df['Altitude / m'])]

# Save to a CSV file.
print(df[:5])

df.to_csv(aircraft_csv_file)
