from distutils.log import error
import os
import numpy as np
import pandas as pd
from   netCDF4 import Dataset
import datetime as datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import iris.pandas

def flight_dictionary() :

    """
    Create a dictionary of flight information including the flight number,
    flight date, start and end times of the flight.
    """

    flight_dict = {'M270' : {'flight_date' : '20200915',
                             'start_time'  : '12:13:00',
                             'end_time'    : '13:38:00'},
                   'M296' : {'flight_date' : '20210701',
                             'start_time'  : '11:23:06',
                             'end_time'    : '12:52:52'},
                   'M302' : {'flight_date' : '20210722',
                             'start_time'  : '11:04:47',
                             'end_time'    : '12:11:11'},
                   'C110' : {'flight_date' : '20180629',
                             'start_time'  : '09:32:56',
                             'end_time'    : '14:30:04'}}

    return flight_dict

def species_dictionary(flight_number) :

    """
    Create a dictionary of species information for creating the figures.
    This information includes the species codes as used in the data files,
    labels to use for figure axes and unit conversions to convert between
    different units.
    """
    if flight_number in {'C110'}:
        species_dict = {'PM2.5' : {'code':  'PM2.5 / ug m-3',
                                'label': 'PM$_2$$_.$$_5$ / \u03BCg m$^-$$^3$',
                                'column_key': 'total_pm25_concentration',
                                'unit_conv': 1}}

    elif flight_number in {'M270', 'M296', 'M302'}:
        species_dict = {'NO2'     : {'code':  'NO2 / ug m-3',
                                    'label': 'NO$_2$ / \u03BCg m$^-$$^3$',
                                    'column_key': 'mass_fraction_of_nitrogen_dioxide_in_air',
                                    'unit_conv': 1.913e9},
                        'O3'      : {'code':  'O3 / ug m-3',
                                    'label': 'O$_3$ / \u03BCg m$^-$$^3$',
                                    'column_key': 'mass_fraction_of_ozone_in_air',
                                    'unit_conv': 1.996e9},
                        'SO2'     : {'code':  'SO2 / ug m-3',
                                    'label': 'SO$_2$ / \u03BCg m$^-$$^3$',
                                    'column_key': 'mass_fraction_of_sulfur_dioxide_expressed_as_sulfur_in_air',
                                    'unit_conv': 2.661e9},
                        'NO2_ppb' : {'code':  'NO2 / ppb',
                                    'label': 'NO$_2$ / ppb',
                                    'column_key': 'mass_fraction_of_nitrogen_dioxide_in_air',
                                    'unit_conv': 1e9},
                        'O3_ppb'  : {'code':  'O3 / ppb',
                                    'label': 'O$_3$ / ppb',
                                    'column_key': 'mass_fraction_of_ozone_in_air',
                                    'unit_conv': 1e9},
                        'SO2_ppb' : {'code':  'SO2 / ppb',
                                    'label': 'SO$_2$ / ppb',
                                    'column_key': 'mass_fraction_of_sulfur_dioxide_expressed_as_sulfur_in_air',
                                    'unit_conv': 1e9},
                        'PM2.5'   : {'code':  'PM2.5 / ug m-3',
                                    'label': 'PM$_2$$_.$$_5$ / \u03BCg m$^-$$^3$',
                                    'column_key': 'mass_concentration_of_pm2p5_dry_aerosol_in_air',
                                    'unit_conv': 1}}

    return species_dict

def read_data(datadir,datafile) :

    """
    Read the data file into a data frame.
    """

    df = pd.read_csv(datadir+datafile,index_col=0)
    df.index = pd.to_datetime(df.index)

    return df

def combine_data(setup,code) :

    """
    Combine the model and aircraft data into a single data frame.
    Remove any model data points without corresponding observations.
    """

    time_filter_flag, start_time, end_time, data = setup[:4]

    df = pd.DataFrame()

    aircraft_df = data['aircraft']

    # Define the coordinate data.
    alt_data = aircraft_df['Altitude / m'][:].tolist()
    lat_data = aircraft_df['Latitude / degrees north'][:].tolist()
    lon_data = aircraft_df['Longitude / degrees east'][:].tolist()
    df['Altitude']  = pd.Series(alt_data,index=aircraft_df.index)
    df['Latitude']  = pd.Series(lat_data,index=aircraft_df.index)
    df['Longitude'] = pd.Series(lon_data,index=aircraft_df.index)

    # Define the aircraft data.
    aircraft_data = aircraft_df[code][:].tolist()
    if '--' in aircraft_data :
        aircraft_data = [str(a).replace('--','nan') for a in aircraft_data]
    aircraft_data = [float(a) for a in aircraft_data]
    df['Aircraft'] = pd.Series(aircraft_data,index=aircraft_df.index)

    # Remove data with no observations.
    df.index = pd.to_datetime(df.index)
    df.resample('1S').mean()
    df = df[np.isfinite(df['Aircraft'])] 

    if "model" in data:
        model_df = data['model']

        # Define the model data.
        model_data  = model_df[code][:].tolist()
        model_data  = [float(m) for m in model_data]
        df['Model'] = pd.Series(model_data,index=model_df.index)
        
        # Add the boundary layer height information to the data frame.
        if 'Boundary Layer Thickness / m' in model_df:
            bl_data = model_df['Boundary Layer Thickness / m'][:].tolist()
            bl_data = [float(b) for b in bl_data]
            df['Boundary_Layer_Height'] = pd.Series(bl_data,index=model_df.index)

        # Add the model wind data to the data frame.
        if 'U Wind / m s-1' and 'V Wind / m s-1' in model_df:
            m_u_data = model_df['U Wind / m s-1'][:].tolist()
            m_u_data = [float(u) for u in m_u_data]
            df['Model_U_Wind'] = pd.Series(m_u_data,index=model_df.index)

            m_v_data = model_df['V Wind / m s-1'][:].tolist()
            m_v_data = [float(v) for v in m_v_data]
            df['Model_V_Wind'] = pd.Series(m_v_data,index=model_df.index)


    # Cut out the desired time interval.
    if time_filter_flag == True :
        df = df.between_time(start_time,end_time)

    return df

def resample_data(df,resample_time,avg_method,min_method,max_method, m_flag) :

    """
    Resample onto the chosen timestep.
    Calculate average, minimum and maximum values.
    """

    # Resample the data frame.
    if avg_method == 'mean' :
        avg_df = df.resample(resample_time).mean()
    if avg_method == 'median' :
        avg_df = df.resample(resample_time).median()
    min_df = df.resample(resample_time).quantile(min_method)
    max_df = df.resample(resample_time).quantile(max_method)

    # Add the range data to the data frame.
    new_df = avg_df

    new_df = new_df.rename(columns={'Aircraft':'Aircraft_Avg'})
    new_df['Aircraft_Min'] = pd.Series(min_df['Aircraft'][:],index=min_df.index)
    new_df['Aircraft_Max'] = pd.Series(max_df['Aircraft'][:],index=max_df.index)

    if (m_flag):
        new_df = new_df.rename(columns={'Model':'Model_Avg'})
        new_df['Model_Min']    = pd.Series(min_df['Model'][:],index=min_df.index)
        new_df['Model_Max']    = pd.Series(max_df['Model'][:],index=max_df.index)

        if 'Boundary_Layer_Height' in new_df:
            new_df = new_df.rename(columns={'Boundary_Layer_Height':'Boundary_Layer_Height_Avg'})
            new_df['Boundary_Layer_Height_Min'] = pd.Series(min_df['Boundary_Layer_Height'][:],index=min_df.index)
            new_df['Boundary_Layer_Height_Max'] = pd.Series(max_df['Boundary_Layer_Height'][:],index=max_df.index)

        if 'Model_U_Wind' and 'Model_V_Wind' in new_df:
            new_df = new_df.rename(columns={'Model_U_Wind':'Model_U_Wind_Avg','Model_V_Wind':'Model_V_Wind_Avg'})

    return new_df

def bin_data(dimension, df, avg_method, min_method, max_method, bin, m_flag):
    """
    Divide the data into bins based on var_data and calculate averages and ranges of the data.
    """
    # Read the data from the data frame.
    var_data = df[dimension][:].tolist()

    # Calculate the minimum and maximum data point.
    min_var = np.nanmin(var_data)
    max_var = np.nanmax(var_data)

    if dimension == 'Altitude':
        start = 0

    elif dimension == 'Latitude':
        start = int(min_var*10) / 10

    elif dimension == 'Longitude':
        if min_var > 0 :
            start = int(min_var*10) / 10
        else :
            start = int((min_var-bin)*10) / 10

    # Create dictionary to hold the processed data.
    data = {} 
    data['binned'] = []

    aircraft_data = df['Aircraft'][:].tolist()
    data['a_avg'] = []
    data['a_min'] = []
    data['a_max'] = []

    if(m_flag):
        model_data = df['Model'][:].tolist()
        data['m_avg'] = []
        data['m_min'] = []
        data['m_max'] = []

    # Loop over each bin and calculate the average and range data.
    while start < max_var :
        end = start + bin
        mid = (start + end) / 2
        data['binned'].append(mid)
        temp_aircraft = []
        if(m_flag): temp_model = []
        for x in range(len(var_data)) :
            if var_data[x] >= start :
                if var_data[x] < end :
                    temp_aircraft.append(aircraft_data[x])
                    if(m_flag): temp_model.append(model_data[x])

        if avg_method == 'mean' :
            data['a_avg'].append(np.nanmean(temp_aircraft))
            if(m_flag): data['m_avg'].append(np.nanmean(temp_model))

        if avg_method == 'median' :
            data['a_avg'].append(np.nanmedian(temp_aircraft))
            if(m_flag): data['m_avg'].append(np.nanmedian(temp_model))

        data['a_min'].append(np.nanpercentile(temp_aircraft,min_method*100))
        data['a_max'].append(np.nanpercentile(temp_aircraft,max_method*100))
        if(m_flag):
            data['m_min'].append(np.nanpercentile(temp_model,min_method*100))
            data['m_max'].append(np.nanpercentile(temp_model,max_method*100))
        start += bin
    return data

def read_data_values(df, m_flag) :

    """
    Read the aircraft and model data from the data frame.
    """
    values = {}

    values['a_min'] = df['Aircraft_Min'][:]
    values['a_avg'] = df['Aircraft_Avg'][:]
    values['a_max'] = df['Aircraft_Max'][:]

    if(m_flag):
        values['m_min'] = df['Model_Min'][:]
        values['m_avg'] = df['Model_Avg'][:]
        values['m_max'] = df['Model_Max'][:]

        if 'Boundary_Layer_Height_Avg' in df:
            values['bl_min'] = df['Boundary_Layer_Height_Min'][:]
            values['bl_avg'] = df['Boundary_Layer_Height_Avg'][:]
            values['bl_max'] = df['Boundary_Layer_Height_Max'][:]

    return values

def resample_wind_data(df,resample_time) :

    """
    Resample onto the chosen timestep.
    Calculate average, minimum and maximum values.
    """

    m_u_data = df['Model_U_Wind'][:].tolist()
    m_v_data = df['Model_V_Wind'][:].tolist()

    speed = [np.sqrt(u**2+v**2) for u,v in zip(m_u_data,m_v_data)]

    # Resample the data frame.
    if resample_time == '10s' :
        new_df = df.iloc[::10,:]
    if resample_time == '30s' :
        new_df = df.iloc[::30,:]
    if resample_time == '60s' :
        new_df = df.iloc[::60,:]

    return new_df

def read_data_values_wind(df) :

    """
    Read the wind data from the data frame.
    """

    m_uwind = df['Model_U_Wind'][:]
    m_vwind = df['Model_V_Wind'][:]

    return m_uwind,m_vwind

def read_cross_section(datadir,datafile,column_key,unit_conv,time_filter_flag,start_time,end_time,flight_date) :

    """
    Read the cross section data from the file.
    """

    f = Dataset(datadir+datafile,mode='r',format='NETCDF4')
    column = {}
    column['time'] = pd.to_datetime(f['time'][:],unit='h')
    column['alt']  = f['level_height'][:]
    column['data'] = f[column_key][:,:].T
    column['data'] = column['data'] * unit_conv

    if time_filter_flag == True :
        start_datetime = pd.to_datetime(datetime.datetime(int(flight_date[0:4]),int(flight_date[4:6]),int(flight_date[6:8]),int(start_time[0:2]),int(start_time[3:5]),int(start_time[6:8])))
        end_datetime   = pd.to_datetime(datetime.datetime(int(flight_date[0:4]),int(flight_date[4:6]),int(flight_date[6:8]),int(end_time[0:2]),int(end_time[3:5]),int(end_time[6:8])))
        start_index = 'None'
        end_index   = 'None'
        for x in range(len(column['time'])-1) :
            if column['time'][x] <= start_datetime and column['time'][x+1] > start_datetime :
                start_index = x
            if column['time'][x] <= end_datetime and column['time'][x+1] > end_datetime :
                end_index = x
        if start_index != 'None' and end_index != 'None' :
            column['time'] = column['time'][start_index:end_index+1]
            column['data'] = column['data'][:,start_index:end_index+1]
        elif start_index != 'None' and end_index == 'None' :
            if end_datetime > column['time'][-1] :
                column['time'] = column['time'][start_index:]
                column['data'] = column['data'][:,start_index:]
            else :
                print('Error with end time index')
        else :
            print('Error with start time index')

    return column

def setup_figure() :

    """
    Set up the figure.
    """

    fig = plt.figure(figsize=(15,15))
    ax  = plt.axes([.2,.15,.75,.7])

    return fig, ax

def setup_map(fig,n,m_flag) :

    """
    Set up the map.
    """
    if(m_flag): i = 3
    else: i = 1

    ax = fig.add_subplot(1,i,n,projection=ccrs.PlateCarree())

    ax.outline_patch.set_linewidth(5)

    ax.background_patch.set_visible(False)

    land = cfeature.NaturalEarthFeature('physical','land','10m',facecolor='dimgrey',alpha=0.5)

    ax.add_feature(land,zorder=1)

    return ax

def calculate_time_markers(time_data) :

    """
    Calculate time markers for each 15 minute interval.
    """

    start_time  = time_data[0]
    end_time    = time_data[-1]
    hour_range  = np.arange(start_time.hour,end_time.hour+1,1).tolist()
    time_ticks  = []
    time_labels = []
    for t in range(len(hour_range)) :
        time_ticks.append(pd.to_datetime(datetime.datetime(start_time.year,start_time.month,
                                                           start_time.day,hour_range[t],0,0)))
        time_ticks.append(pd.to_datetime(datetime.datetime(start_time.year,start_time.month,
                                                           start_time.day,hour_range[t],15,0)))
        time_ticks.append(pd.to_datetime(datetime.datetime(start_time.year,start_time.month,
                                                           start_time.day,hour_range[t],30,0)))
        time_ticks.append(pd.to_datetime(datetime.datetime(start_time.year,start_time.month,
                                                           start_time.day,hour_range[t],45,0)))
        if hour_range[t] < 10 :
            time_labels.append('0'+str(hour_range[t])+':00')
            time_labels.append('0'+str(hour_range[t])+':15')
            time_labels.append('0'+str(hour_range[t])+':30')
            time_labels.append('0'+str(hour_range[t])+':45')
        else :
            time_labels.append(str(hour_range[t])+':00')
            time_labels.append(str(hour_range[t])+':15')
            time_labels.append(str(hour_range[t])+':30')
            time_labels.append(str(hour_range[t])+':45')

    for x in range(len(time_ticks)) :
        if time_ticks[x] > start_time :
            start_index = x
            break
    for x in range(len(time_ticks)) :
        if time_ticks[x] > end_time :
            end_index = x
            break
    if end_time > time_ticks[-1] :
        time_ticks  = time_ticks[start_index:]
        time_labels = time_labels[start_index:]
    else :
        time_ticks  = time_ticks[start_index:end_index]
        time_labels = time_labels[start_index:end_index]

    return time_ticks, time_labels

def setup_notebook(flight_number, m_flag) :

    # Extract the flight information from the dictionary.
    flight_dict = flight_dictionary()
    if(flight_number not in flight_dict):
        raise Exception('Flight ' + flight_number + ' does not exist.')

    flight_date = flight_dict[flight_number]['flight_date']
    start_time  = flight_dict[flight_number]['start_time']
    end_time    = flight_dict[flight_number]['end_time']

    # Define the suite name.
    # For model run variations of the same flight.
    suite = 'mi-bd327'

    # Define the plot directory.
    # This doesn't need changing unless you want to save the plots to a different location.
    plotdir = './Plots/'+flight_number+'/'+suite+'/'

    # Define the time interval information.
    # Set the time_filter_flag to True in order to cut out the chosen time range.
    # Set the time_filter_flag to False in order to use the whole flight.
    # The start_time and end_time are defined in the flight dictionary.
    # They can be overwritten with different values here if required.
    time_filter_flag = False
    start_time = 'HH:MM:SS'
    end_time   = 'HH:MM:SS'

    data = {}
  
    obsdir = './Data_Files/Aircraft/'+flight_number+'/'
    aircraft_track_file = flight_number + '_' + flight_date + '_Aircraft_Track_Data.csv'
    aircraft_df = read_data(obsdir,aircraft_track_file)
    data['aircraft'] = aircraft_df
    
    if(m_flag):
        modeldir = './Data_Files/Model/'+flight_number+'/'+suite+'/'  # Define the data directory.
        model_track_file    = flight_number + '_' + flight_date + '_Model_Track_Data.csv'
        model_column_file   = flight_number + '_' + flight_date + '_Model_Column_Data.nc'# Define the data files.
        model_df    = read_data(modeldir,model_track_file)   # Read the data.
        data['model'] = model_df
    else:
        modeldir = None
        model_column_file = None


    options = {
        # Define the resampling time and averaging methods.
        # The avg_method can be either 'mean' or 'median'.
        # The min_method and max_method refer to the quantile of the data.
        # For the minimum and maximum select 0 and 1, for the interquartile range, select 0.25 and 0.75.
        'avg_method': 'mean',
        'resample_time': '60s',
        'min_method': 0,
        'max_method': 1,
        # Define the bin sizes for altitude (in metres), latitude (in degrees north) and longitude (in degrees east).
        'alt_bin': 50,
        'lat_bin': 0.1,
        'lon_bin': 0.1,
        # Define the colours to be used for the datasets.
        'a_colour': 'darkslategrey', #aircraft colour
        'm_colour': 'indianred' #model colour
    }

    return time_filter_flag, start_time, end_time, data, options, plotdir, modeldir, model_column_file, flight_date