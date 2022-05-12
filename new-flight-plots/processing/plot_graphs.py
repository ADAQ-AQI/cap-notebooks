import os
import numpy as np
import pandas as pd
from   netCDF4 import Dataset
import datetime as datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

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
                             'end_time'    : '12:11:11'}}

    return flight_dict

def species_dictionary() :

    """
    Create a dictionary of species information for creating the figures.
    This information includes the species codes as used in the data files,
    labels to use for figure axes and unit conversions to convert between
    different units.
    """

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

def combine_data(aircraft_df,model_df,code,time_filter_flag,start_time,end_time) :

    """
    Combine the model and aircraft data into a single data frame.
    Remove any model data points without corresponding observations.
    """

    df = pd.DataFrame()

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
    aircraft_data = [np.float(a) for a in aircraft_data]
    df['Aircraft'] = pd.Series(aircraft_data,index=aircraft_df.index)

    # Define the model data.
    model_data  = model_df[code][:].tolist()
    model_data  = [np.float(m) for m in model_data]
    df['Model'] = pd.Series(model_data,index=model_df.index)

    # Add the boundary layer height information to the data frame.
    bl_data = model_df['Boundary Layer Thickness / m'][:].tolist()
    bl_data = [np.float(b) for b in bl_data]
    df['Boundary_Layer_Height'] = pd.Series(bl_data,index=model_df.index)

    # Add the model wind data to the data frame.
    m_u_data = model_df['U Wind / m s-1'][:].tolist()
    m_v_data = model_df['V Wind / m s-1'][:].tolist()
    m_u_data = [np.float(u) for u in m_u_data]
    m_v_data = [np.float(v) for v in m_v_data]
    df['Model_U_Wind'] = pd.Series(m_u_data,index=model_df.index)
    df['Model_V_Wind'] = pd.Series(m_v_data,index=model_df.index)

    # Remove data with no observations.
    df.index = pd.to_datetime(df.index)
    df.resample('1S').mean()
    df = df[np.isfinite(df['Aircraft'])]

    # Cut out the desired time interval.
    if time_filter_flag == True :
        df = df.between_time(start_time,end_time)

    return df

def resample_data(df,resample_time,avg_method,min_method,max_method) :

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
    new_df = avg_df.rename(columns={'Aircraft':'Aircraft_Avg','Model':'Model_Avg','Boundary_Layer_Height':'Boundary_Layer_Height_Avg',
                                    'Model_U_Wind':'Model_U_Wind_Avg','Model_V_Wind':'Model_V_Wind_Avg'})
    new_df['Aircraft_Min'] = pd.Series(min_df['Aircraft'][:],index=min_df.index)
    new_df['Model_Min']    = pd.Series(min_df['Model'][:],index=min_df.index)
    new_df['Boundary_Layer_Height_Min'] = pd.Series(min_df['Boundary_Layer_Height'][:],index=min_df.index)
    new_df['Aircraft_Max'] = pd.Series(max_df['Aircraft'][:],index=max_df.index)
    new_df['Model_Max']    = pd.Series(max_df['Model'][:],index=max_df.index)
    new_df['Boundary_Layer_Height_Max'] = pd.Series(max_df['Boundary_Layer_Height'][:],index=max_df.index)

    return new_df

def bin_altitude_data(df,avg_method,min_method,max_method,alt_bin) :

    """
    Divide the data into bins based on altitude and calculate averages and ranges of the data.
    """

    # Read the data from the data frame.
    alt_data      = df['Altitude'][:].tolist()
    model_data    = df['Model'][:].tolist()
    aircraft_data = df['Aircraft'][:].tolist()

    # Calculate the maximum altitude data point.
    max_alt = np.nanmax(alt_data)

    # Create lists to hold the processed data.
    binned_alt = []
    m_avg = []
    m_min = []
    m_max = []
    a_avg = []
    a_min = []
    a_max = []

    # Loop over each bin and calculate the average and range data.
    start = 0
    while start < max_alt :
        end = start + alt_bin
        mid = (start + end) / 2
        binned_alt.append(mid)
        temp_model = []
        temp_aircraft = []
        for x in range(len(alt_data)) :
            if alt_data[x] >= start :
                if alt_data[x] < end :
                    temp_model.append(model_data[x])
                    temp_aircraft.append(aircraft_data[x])
        if avg_method == 'mean' :
            a_avg.append(np.nanmean(temp_aircraft))
            m_avg.append(np.nanmean(temp_model))
        if avg_method == 'median' :
            a_avg.append(np.nanmedian(temp_aircraft))
            m_avg.append(np.nanmedian(temp_model))
        a_min.append(np.nanpercentile(temp_aircraft,min_method*100))
        a_max.append(np.nanpercentile(temp_aircraft,max_method*100))
        m_min.append(np.nanpercentile(temp_model,min_method*100))
        m_max.append(np.nanpercentile(temp_model,max_method*100))
        start += alt_bin

    return binned_alt, a_min, a_avg, a_max, m_min, m_avg, m_max

def bin_latitude_data(df,avg_method,min_method,max_method,lat_bin) :

    """
    Divide the data into bins based on latitude and calculate averages and ranges of the data.
    """

    # Read the data from the data frame.
    lat_data      = df['Latitude'][:].tolist()
    model_data    = df['Model'][:].tolist()
    aircraft_data = df['Aircraft'][:].tolist()

    # Calculate the minimum and maximum latitude data point.
    min_lat = np.nanmin(lat_data)
    max_lat = np.nanmax(lat_data)

    # Create lists to hold the processed data.
    binned_lat = []
    m_avg = []
    m_min = []
    m_max = []
    a_avg = []
    a_min = []
    a_max = []

    # Loop over each bin and calculate the average and range data.
    start = int(min_lat*10) / 10
    end_lat = int((max_lat+lat_bin)*10) / 10
    while start < max_lat :
        end = start + lat_bin
        mid = (start + end) / 2
        binned_lat.append(mid)
        temp_model = []
        temp_aircraft = []
        for x in range(len(lat_data)) :
            if lat_data[x] >= start :
                if lat_data[x] < end :
                    temp_model.append(model_data[x])
                    temp_aircraft.append(aircraft_data[x])
        if avg_method == 'mean' :
            a_avg.append(np.nanmean(temp_aircraft))
            m_avg.append(np.nanmean(temp_model))
        if avg_method == 'median' :
            a_avg.append(np.nanmedian(temp_aircraft))
            m_avg.append(np.nanmedian(temp_model))
        a_min.append(np.nanpercentile(temp_aircraft,min_method*100))
        a_max.append(np.nanpercentile(temp_aircraft,max_method*100))
        m_min.append(np.nanpercentile(temp_model,min_method*100))
        m_max.append(np.nanpercentile(temp_model,max_method*100))
        start += lat_bin

    return binned_lat, a_min, a_avg, a_max, m_min, m_avg, m_max

def bin_longitude_data(df,avg_method,min_method,max_method,lon_bin) :

    """
    Divide the data into bins based on longitude and calculate averages and ranges of the data.
    """

    # Read the data from the data frame.
    lon_data      = df['Longitude'][:].tolist()
    model_data    = df['Model'][:].tolist()
    aircraft_data = df['Aircraft'][:].tolist()

    # Calculate the minimum and maximum longitude data point.
    min_lon = np.nanmin(lon_data)
    max_lon = np.nanmax(lon_data)

    # Create lists to hold the processed data.
    binned_lon = []
    m_avg = []
    m_min = []
    m_max = []
    a_avg = []
    a_min = []
    a_max = []

    # Loop over each bin and calculate the average and range data.
    if min_lon > 0 :
        start = int(min_lon*10) / 10
    else :
        start = int((min_lon-lon_bin)*10) / 10
    while start < max_lon :
        end = start + lon_bin
        mid = (start + end) / 2
        binned_lon.append(mid)
        temp_model = []
        temp_aircraft = []
        for x in range(len(lon_data)) :
            if lon_data[x] >= start :
                if lon_data[x] < end :
                    temp_model.append(model_data[x])
                    temp_aircraft.append(aircraft_data[x])
        if avg_method == 'mean' :
            a_avg.append(np.nanmean(temp_aircraft))
            m_avg.append(np.nanmean(temp_model))
        if avg_method == 'median' :
            a_avg.append(np.nanmedian(temp_aircraft))
            m_avg.append(np.nanmedian(temp_model))
        a_min.append(np.nanpercentile(temp_aircraft,min_method*100))
        a_max.append(np.nanpercentile(temp_aircraft,max_method*100))
        m_min.append(np.nanpercentile(temp_model,min_method*100))
        m_max.append(np.nanpercentile(temp_model,max_method*100))
        start += lon_bin

    return binned_lon, a_min, a_avg, a_max, m_min, m_avg, m_max

def read_data_values(df) :

    """
    Read the aircraft and model data from the data frame.
    """

    a_min = df['Aircraft_Min'][:]
    a_avg = df['Aircraft_Avg'][:]
    a_max = df['Aircraft_Max'][:]

    m_min = df['Model_Min'][:]
    m_avg = df['Model_Avg'][:]
    m_max = df['Model_Max'][:]

    bl_min = df['Boundary_Layer_Height_Min'][:]
    bl_avg = df['Boundary_Layer_Height_Avg'][:]
    bl_max = df['Boundary_Layer_Height_Max'][:]

    return a_min, a_avg, a_max, m_min, m_avg, m_max, bl_min, bl_avg, bl_max

def resample_wind_data(df,resample_time) :

    """
    Resample onto the chosen timestep.
    Calculate average, minimum and maximum values.
    """

    m_u_data = df['Model_U_Wind'][:].tolist()
    m_v_data = df['Model_V_Wind'][:].tolist()

    speed = [np.sqrt(u**2+v**2) for u,v in zip(m_u_data,m_v_data)]
    print(np.nanmin(speed),np.nanmax(speed))


    # Resample the data frame.
    if resample_time == '10S' :
        new_df = df.iloc[::10,:]
    if resample_time == '30S' :
        new_df = df.iloc[::30,:]
    if resample_time == '60S' :
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

    column_time = pd.to_datetime(f['time'][:],unit='h')
    column_alt  = f['level_height'][:]
    column_data = f[column_key][:,:].T
    column_data = column_data * unit_conv

    if time_filter_flag == True :
        start_datetime = pd.to_datetime(datetime.datetime(int(flight_date[0:4]),int(flight_date[4:6]),int(flight_date[6:8]),int(start_time[0:2]),int(start_time[3:5]),int(start_time[6:8])))
        end_datetime   = pd.to_datetime(datetime.datetime(int(flight_date[0:4]),int(flight_date[4:6]),int(flight_date[6:8]),int(end_time[0:2]),int(end_time[3:5]),int(end_time[6:8])))
        start_index = 'None'
        end_index   = 'None'
        for x in range(len(column_time)-1) :
            if column_time[x] <= start_datetime and column_time[x+1] > start_datetime :
                start_index = x
            if column_time[x] <= end_datetime and column_time[x+1] > end_datetime :
                end_index = x
        if start_index is not 'None' and end_index is not 'None' :
            column_time = column_time[start_index:end_index+1]
            column_data = column_data[:,start_index:end_index+1]
        elif start_index is not 'None' and end_index is 'None' :
            if end_datetime > column_time[-1] :
                column_time = column_time[start_index:]
                column_data = column_data[:,start_index:]
            else :
                print('Error with end time index')
        else :
            print('Error with start time index')

    return column_time, column_alt, column_data

def setup_figure() :

    """
    Set up the figure.
    """

    fig = plt.figure(figsize=(15,15))
    ax  = plt.axes([.2,.15,.75,.7])

    return fig, ax

def setup_map() :

    """
    Set up the map.
    """

    bkmap = cimgt.GoogleTiles(style='street')

    fig = plt.figure(figsize=(15,15))
    ax = plt.axes([.1,.05,.8,.8],projection=bkmap.crs)
    ax.outline_patch.set_linewidth(5)
    ax.background_patch.set_visible(False)

    ax.add_image(bkmap, 10) #, interpolation='spline36')

    return fig, ax

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

def plot_timeseries(df,plotdir,key,label,a_colour,m_colour) :

    """
    Plot timeseries comparisons of aircraft and model data.
    """

    # Define the figure.
    fig,ax = setup_figure()

    # Define the data.
    time_data = df.index
    a_min, a_avg, a_max, m_min, m_avg, m_max, bl_min, bl_avg, bl_max = read_data_values(df)

    # Plot the data.
    plt.plot(time_data,a_avg,lw=5,c=a_colour,label='Aircraft')
    plt.fill_between(time_data,a_min,a_max,fc=a_colour,ec=None,alpha=0.5)
    plt.plot(time_data,m_avg,lw=5,c=m_colour,label='Model')
    plt.fill_between(time_data,m_min,m_max,fc=m_colour,ec=None,alpha=0.5)

    # Set the axes labels.
    plt.xlabel('Time / UTC',fontsize=50,labelpad=10)
    plt.ylabel(label,fontsize=50,labelpad=10)

    # Set the axes ticks.
    time_ticks,time_labels = calculate_time_markers(time_data)
    plt.xticks(time_ticks)
    ax.set_xticklabels(time_labels)
    ax.tick_params(labelsize=30)

    # Add a legend.
    plt.legend(fontsize=30,markerscale=2)

    # Save the figure.
    plt.savefig(plotdir+key+'_Timeseries.png')
    plt.close()

def plot_altitude_profile(full_df,df,plotdir,key,label,a_colour,m_colour,alt_bin,avg_method,min_method,max_method) :

    """
    Plot altitude profile comparisons of aircraft and model data.
    """

    # Define the figure.
    fig,ax = setup_figure()

    # Define the data.
    alt_data = df['Altitude']
    a_min, a_avg, a_max, m_min, m_avg, m_max, bl_min, bl_avg, bl_max = read_data_values(df)

    # Plot the data.
    plt.scatter(a_avg,alt_data,s=200,c=a_colour,label='Aircraft')
    plt.scatter(m_avg,alt_data,s=200,c=m_colour,label='Model')

    # Set the axes labels.
    plt.xlabel(label,fontsize=50,labelpad=10)
    plt.ylabel('Altitude / m',fontsize=50,labelpad=10)

    # Set the axes ticks.
    ax.tick_params(labelsize=30)

    # Add a legend.
    plt.legend(fontsize=30,markerscale=2)

    # Save the figure.
    plt.savefig(plotdir+key+'_Altitude_Profile.png')
    plt.close()

    """
    Plot altitude profile comparisons of aircraft and model data with binned data.
    """

    # Define the figure.
    fig,ax = setup_figure()

    # Define the data.
    binned_alt,a_min,a_avg,a_max,m_min,m_avg,m_max = bin_altitude_data(full_df,avg_method,min_method,max_method,alt_bin)

    # Plot the data.
    plt.plot(a_avg,binned_alt,lw=5,c=a_colour,label='Aircraft')
    plt.fill_betweenx(binned_alt,a_min,a_max,fc=a_colour,ec=None,alpha=0.5)
    plt.plot(m_avg,binned_alt,lw=5,c=m_colour,label='Model')
    plt.fill_betweenx(binned_alt,m_min,m_max,fc=m_colour,ec=None,alpha=0.5)

    # Set the axes labels.
    plt.xlabel(label,fontsize=50,labelpad=10)
    plt.ylabel('Altitude / m',fontsize=50,labelpad=10)

    # Set the axes ticks.
    ax.tick_params(labelsize=30)

    # Add a legend.
    plt.legend(fontsize=30,markerscale=2)

    # Save the figure.
    plt.savefig(plotdir+key+'_Altitude_Profile_Binned.png')
    plt.close()

def plot_latitude_profile(full_df,df,plotdir,key,label,a_colour,m_colour,lat_bin,avg_method,min_method,max_method) :

    """
    Plot latitude profile comparisons of aircraft and model data.
    """

    # Define the figure.
    fig,ax = setup_figure()

    # Define the data.
    lat_data = df['Latitude']
    a_min, a_avg, a_max, m_min, m_avg, m_max, bl_min, bl_avg, bl_max = read_data_values(df)

    # Plot the data.
    plt.scatter(lat_data,a_avg,s=200,c=a_colour,label='Aircraft')
    plt.scatter(lat_data,m_avg,s=200,c=m_colour,label='Model')

    # Set the axes labels.
    plt.xlabel('Latitude / degrees north',fontsize=50,labelpad=10)
    plt.ylabel(label,fontsize=50,labelpad=10)

    # Set the axes ticks.
    ax.tick_params(labelsize=30)

    # Add a legend.
    plt.legend(fontsize=30,markerscale=2)

    # Save the figure.
    plt.savefig(plotdir+key+'_Latitude_Profile.png')
    plt.close()

    """
    Plot latitude profile comparisons of aircraft and model data with binned data.
    """

    # Define the figure.
    fig,ax = setup_figure()

    # Define the data.
    binned_lat,a_min,a_avg,a_max,m_min,m_avg,m_max = bin_latitude_data(full_df,avg_method,min_method,max_method,lat_bin)

    # Plot the data.
    plt.plot(binned_lat,a_avg,lw=5,c=a_colour,label='Aircraft')
    plt.fill_between(binned_lat,a_min,a_max,fc=a_colour,ec=None,alpha=0.5)
    plt.plot(binned_lat,m_avg,lw=5,c=m_colour,label='Model')
    plt.fill_between(binned_lat,m_min,m_max,fc=m_colour,ec=None,alpha=0.5)

    # Set the axes labels.
    plt.xlabel('Latitude / degrees north',fontsize=50,labelpad=10)
    plt.ylabel(label,fontsize=50,labelpad=10)

    # Set the axes ticks.
    ax.tick_params(labelsize=30)

    # Add a legend.
    plt.legend(fontsize=30,markerscale=2)

    # Save the figure.
    plt.savefig(plotdir+key+'_Latitude_Profile_Binned.png')
    plt.close()

def plot_longitude_profile(full_df,df,plotdir,key,label,a_colour,m_colour,lon_bin,avg_method,min_method,max_method) :

    """
    Plot longitude profile comparisons of aircraft and model data.
    """

    # Define the figure.
    fig,ax = setup_figure()

    # Define the data.
    lat_data = df['Longitude']
    a_min, a_avg, a_max, m_min, m_avg, m_max, bl_min, bl_avg, bl_max = read_data_values(df)

    # Plot the data.
    plt.scatter(lat_data,a_avg,s=200,c=a_colour,label='Aircraft')
    plt.scatter(lat_data,m_avg,s=200,c=m_colour,label='Model')

    # Set the axes labels.
    plt.xlabel('Longitude / degrees east',fontsize=50,labelpad=10)
    plt.ylabel(label,fontsize=50,labelpad=10)

    # Set the axes ticks.
    ax.tick_params(labelsize=30)

    # Add a legend.
    plt.legend(fontsize=30,markerscale=2)

    # Save the figure.
    plt.savefig(plotdir+key+'_Longitude_Profile.png')
    plt.close()

    """
    Plot latitude profile comparisons of aircraft and model data with binned data.
    """

    # Define the figure.
    fig,ax = setup_figure()

    # Define the data.
    binned_lon,a_min,a_avg,a_max,m_min,m_avg,m_max = bin_longitude_data(full_df,avg_method,min_method,max_method,lon_bin)

    # Plot the data.
    plt.plot(binned_lon,a_avg,lw=5,c=a_colour,label='Aircraft')
    plt.fill_between(binned_lon,a_min,a_max,fc=a_colour,ec=None,alpha=0.5)
    plt.plot(binned_lon,m_avg,lw=5,c=m_colour,label='Model')
    plt.fill_between(binned_lon,m_min,m_max,fc=m_colour,ec=None,alpha=0.5)

    # Set the axes labels.
    plt.xlabel('Longitude / degrees east',fontsize=50,labelpad=10)
    plt.ylabel(label,fontsize=50,labelpad=10)

    # Set the axes ticks.
    ax.tick_params(labelsize=30)

    # Add a legend.
    plt.legend(fontsize=30,markerscale=2)

    # Save the figure.
    plt.savefig(plotdir+key+'_Longitude_Profile_Binned.png')
    plt.close()

def plot_correlations(df,plotdir,key,label) :

    """
    Plot correlation comparisons of aircraft and model data.
    """

    # Define the figure.
    fig,ax = setup_figure()

    # Define the data.
    alt_data = df['Altitude'][:]
    a_min, a_avg, a_max, m_min, m_avg, m_max, bl_min, bl_avg, bl_max = read_data_values(df)

    # Plot the data.
    plt.scatter(a_avg,m_avg,s=200,c=alt_data)

    # Plot a 1:1 agreement line.
    min = np.nanmin([np.nanmin(a_avg),np.nanmin(m_avg)])
    max = np.nanmax([np.nanmax(a_avg),np.nanmax(m_avg)])
    plt.plot([min,max],[min,max],lw=5,ls='dashed',c='grey')

##    # Add a shaded region represention aircraft data below a value of 1.
##    # This is used to represent SO2 data below 1ppb which is below the limit of detection.
##    ymin,ymax = ax.get_ylim()
##    xmin,xmax = ax.get_xlim()
##    plt.fill_betweenx([ymin,ymax],[xmin,xmin],[1,1],ec=None,fc='black',alpha=0.2)

    # Set the axes labels.
    plt.xlabel('Aircraft '+label,fontsize=50,labelpad=10)
    plt.ylabel('Model '+label,fontsize=50,labelpad=10)

    # Set the axes ticks.
    ax.tick_params(labelsize=30)

    # Plot the colourbar.
    cb = plt.colorbar(ax=ax,orientation='vertical',shrink=0.8,pad=0.05,extend='both')
    cb.ax.tick_params(labelsize=25)
    cb.set_label('Altitude / m',fontsize=30,labelpad=10)

    # Save the figure.
    plt.savefig(plotdir+key+'_Correlation.png')
    plt.close()

def plot_maps(df,wind_df,plotdir,key,label) :

    """
    Plot maps of aircraft track.
    """

    # Define the figure.
    fig,ax = setup_map()

    # Define the data.
    lat_data = df['Latitude'][:]
    lon_data = df['Longitude'][:]
    alt_data = df['Altitude'][:]

    # Plot the data.
    plt.scatter(lon_data,lat_data,s=200,c=alt_data,cmap=plt.cm.summer,ec='black',zorder=3,transform=ccrs.PlateCarree())

    # Plot the colourbar.
    cb = plt.colorbar(ax=ax,orientation='horizontal',shrink=0.8,pad=0.05,extend='both')
    cb.ax.tick_params(labelsize=20)
    cb.set_label('Altitude / m',fontsize=20,labelpad=10)

    plt.show()
    # Save the figure.
    #plt.savefig(plotdir+key+'_Map_Track.png')
    #plt.close()

    """
    Plot maps of aircraft data.
    """

    # Define the figure.
    fig,ax = setup_map()

    # Define the data.
    lat_data = df['Latitude'][:]
    lon_data = df['Longitude'][:]
    alt_data = df['Altitude'][:]
    a_min, a_avg, a_max, m_min, m_avg, m_max, bl_min, bl_avg, bl_max = read_data_values(df)

    # Determine the minimum and maximum values.
    min = np.nanmin([np.nanmin(a_avg),np.nanmin(m_avg)])
    max = np.nanmax([np.nanmax(a_avg),np.nanmax(m_avg)])

    # Plot the data.
    plt.scatter(lon_data,lat_data,s=200,c=a_avg,cmap=plt.cm.Spectral_r,ec='black',vmin=min,vmax=max,zorder=3,transform=ccrs.PlateCarree())

    # Plot the colourbar.
    cb = plt.colorbar(ax=ax,orientation='horizontal',shrink=0.8,pad=0.05,extend='both')
    cb.ax.tick_params(labelsize=20)
    cb.set_label(label,fontsize=20,labelpad=10)

    # Save the figure.
    plt.savefig(plotdir+key+'_Map_Aircraft.png')
    plt.close()

    """
    Plot maps of model data.
    """

    # Define the figure.
    fig,ax = setup_map()

    # Define the data.
    lat_data = df['Latitude'][:]
    lon_data = df['Longitude'][:]
    alt_data = df['Altitude'][:]
    a_min, a_avg, a_max, m_min, m_avg, m_max, bl_min, bl_avg, bl_max = read_data_values(df)

    # Determine the minimum and maximum values.
    min = np.nanmin([np.nanmin(a_avg),np.nanmin(m_avg)])
    max = np.nanmax([np.nanmax(a_avg),np.nanmax(m_avg)])

    # Plot the data.
    plt.scatter(lon_data,lat_data,s=200,c=m_avg,cmap=plt.cm.Spectral_r,ec='black',vmin=min,vmax=max,zorder=3,transform=ccrs.PlateCarree())

    # Plot the colourbar.
    cb = plt.colorbar(ax=ax,orientation='horizontal',shrink=0.8,pad=0.05,extend='both')
    cb.ax.tick_params(labelsize=20)
    cb.set_label(label,fontsize=20,labelpad=10)

    # Save the figure.
    plt.savefig(plotdir+key+'_Map_Model.png')
    plt.close()

    """
    Plot maps of model - observation difference data.
    """

    # Define the figure.
    fig,ax = setup_map()

    # Define the data.
    lat_data = df['Latitude'][:]
    lon_data = df['Longitude'][:]
    alt_data = df['Altitude'][:]
    a_min, a_avg, a_max, m_min, m_avg, m_max, bl_min, bl_avg, bl_max = read_data_values(df)

    # Calculate the difference between model and observations.
    diff_data = [x-y for x,y in zip(m_avg,a_avg)]

    # Determine the min and max.
    min = np.nanmin(diff_data)
    max = np.nanmax(diff_data)
    limits = np.sqrt(np.nanmax([min**2,max**2]))
    min_lim = limits*-1
    max_lim = limits

    # Plot the data.
    plt.scatter(lon_data,lat_data,s=200,c=diff_data,cmap=plt.cm.RdBu_r,ec='black',zorder=3,vmin=min_lim,vmax=max_lim,transform=ccrs.PlateCarree())

    # Plot the colourbar.
    cb = plt.colorbar(ax=ax,orientation='horizontal',shrink=0.8,pad=0.05,extend='both')
    cb.ax.tick_params(labelsize=20)
    cb.set_label(label+' (Model - Observation)',fontsize=20,labelpad=10)

    # Save the figure.
    plt.savefig(plotdir+key+'_Map_Difference.png')
    plt.close()
##
##    """
##    Plot the model wind data.
##    """
##
##    fig = plt.figure(figsize=(15,15))
##    ax = plt.axes([.1,.05,.8,.8],projection=ccrs.PlateCarree())
##    ax.outline_patch.set_linewidth(5)
##    ax.background_patch.set_visible(False)
##
###    land = cfeature.NaturalEarthFeature('physical','land','10m',facecolor='dimgrey')
###    ax.add_feature(land,zorder=1)
##    ax.coastlines(resolution='10m',linewidth=3,color='black',zorder=2)
##
##    # Define the figure.
###    fig,ax = setup_map()
##
##    lat_data = wind_df['Latitude'][:]
##    lon_data = wind_df['Longitude'][:]
##    m_uwind,m_vwind = read_data_values_wind(wind_df)
##
##    m_uwind = m_uwind*10
##    m_vwind = m_vwind*10
##
##    plt.barbs(lon_data,lat_data,m_uwind,m_vwind)
##
##    plt.savefig(plotdir+key+'_Model_Wind.png',transparent=True)
##    plt.close()

def plot_cross_sections(df,plotdir,key,label,column_time,column_alt,column_data) :

    """
    Plot cross section comparisons of the model and aircraft data
    """

    # Define the figure.
    fig = plt.figure(figsize=(15,15))
    ax  = plt.axes([.2,.05,.75,.8])

    # Define the data.
    time_data = df.index
    alt_data = df['Altitude'][:]
    a_min, a_avg, a_max, m_min, m_avg, m_max, bl_min, bl_avg, bl_max = read_data_values(df)

    # Calculate the minimum and maximum limits.
    min = np.nanmin([np.nanmin(a_avg),np.nanmin(column_data)])
    max = np.nanmax([np.nanmax(a_avg),np.nanmax(column_data)])

    # Plot the model data.
    plt.pcolormesh(column_time,column_alt,column_data,zorder=1,cmap=plt.cm.Spectral_r,vmin=min,vmax=max)

    # Plot the aircraft data.
    plt.scatter(time_data,alt_data,s=50,c=a_avg,cmap=plt.cm.Spectral_r,edgecolors='black',zorder=2,vmin=min,vmax=max)

    # Plot the boundary layer height.
    plt.plot(time_data,bl_avg,lw=2,ls='dashed',c='black',zorder=3,label='Modelled Boundary Layer Height')

    # Add a legend.
    plt.legend(fontsize=30,markerscale=2)

    # Plot the colourbar.
    cb = plt.colorbar(ax=ax,orientation='horizontal',shrink=0.8,pad=0.25,extend='both')
    cb.ax.tick_params(labelsize=30)
    cb.set_label(label,fontsize=30,labelpad=10)

    # Set the axes ticks.
    time_ticks,time_labels = calculate_time_markers(time_data)
    plt.xticks(time_ticks,rotation=45)
    ax.set_xticklabels(time_labels)
    ax.tick_params(labelsize=30)

    # Set the axes labels.
    plt.xlabel('Time / UTC',fontsize=50,labelpad=10)
    plt.ylabel('Altitude / m',fontsize=50,labelpad=10)

    # Save the figure.
    plt.savefig(plotdir+key+'_Cross_Section.png')
    plt.close()

def plot_curtain(df,plotdir,key,label) :

    """
    Plot a latitude curtain plot
    """

    # Define the figure.
    fix,ax = plt.subplots(1,3,sharex='all',sharey='all',figsize=(25,15))

    # Define the data.
    alt_data = df['Altitude'][:]
    lat_data = df['Latitude'][:]
    a_min, a_avg, a_max, m_min, m_avg, m_max, bl_min, bl_avg, bl_max = read_data_values(df)

    # Calculate the minimum and maximum limits.
    min = np.nanmin([np.nanmin(a_avg),np.nanmin(m_avg)])
    max = np.nanmax([np.nanmax(a_avg),np.nanmax(m_avg)])

    # Calculate the difference between model and observations.
    diff_data = [x-y for x,y in zip(m_avg,a_avg)]

    # Determine the min and max.
    diff_min = np.nanmin(diff_data)
    diff_max = np.nanmax(diff_data)
    limits = np.sqrt(np.nanmax([diff_min**2,diff_max**2]))
    diff_min_lim = limits*-1
    diff_max_lim = limits

    # Plot the aircraft data.
    plt0 = ax[0].scatter(lat_data,alt_data,c=a_avg,marker='X',s=100,cmap=plt.cm.Spectral_r,vmin=min,vmax=max)
    ax[0].set_ylabel('Altitude / m',fontsize=40,labelpad=20)
    ax[0].tick_params(labelsize=30)
    ax[0].grid()
    ax[0].set_title('Aircraft',fontsize=40)
    cb0 = plt.colorbar(plt0,ax=ax[0],orientation='vertical',pad=0.1,extend='both')
    cb0.set_label(label,fontsize=30,labelpad=10)
    cb0.ax.tick_params(labelsize=20)

    # Plot the model data.
    plt1 = ax[1].scatter(lat_data,alt_data,c=m_avg,marker='X',s=100,cmap=plt.cm.Spectral_r,vmin=min,vmax=max)
    ax[1].set_xlabel('Latitude / degrees north',fontsize=40,labelpad=20)
    ax[1].tick_params(labelsize=30)
    ax[1].grid()
    ax[1].set_title('Model',fontsize=40)
    cb1 = plt.colorbar(plt1,ax=ax[1],orientation='vertical',pad=0.1,extend='both')
    cb1.set_label(label,fontsize=30,labelpad=10)
    cb1.ax.tick_params(labelsize=20)

    # Plot the difference data.
    plt2 = ax[2].scatter(lat_data,alt_data,c=diff_data,marker='X',s=100,cmap=plt.cm.RdBu_r,vmin=diff_min_lim,vmax=diff_max_lim)
    ax[2].tick_params(labelsize=30)
    ax[2].grid()
    ax[2].set_title('Difference',fontsize=40)
    cb2 = plt.colorbar(plt2,ax=ax[2],orientation='vertical',pad=0.1,extend='both')
    cb2.set_label(label+' (Model - Observation)',fontsize=30,labelpad=10)
    cb2.ax.tick_params(labelsize=20)

    # Save the figure.
    plt.savefig(plotdir+key+'_Latitude_Curtain_Plot.png')
    plt.close()

    """
    Plot a longitude curtain plot
    """

    # Define the figure.
    fix,ax = plt.subplots(1,3,sharex='all',sharey='all',figsize=(25,15))

    # Define the data.
    alt_data = df['Altitude'][:]
    lon_data = df['Longitude'][:]
    a_min, a_avg, a_max, m_min, m_avg, m_max, bl_min, bl_avg, bl_max = read_data_values(df)

    # Calculate the minimum and maximum limits.
    min = np.nanmin([np.nanmin(a_avg),np.nanmin(m_avg)])
    max = np.nanmax([np.nanmax(a_avg),np.nanmax(m_avg)])

    # Calculate the difference between model and observations.
    diff_data = [x-y for x,y in zip(m_avg,a_avg)]

    # Determine the min and max.
    diff_min = np.nanmin(diff_data)
    diff_max = np.nanmax(diff_data)
    limits = np.sqrt(np.nanmax([diff_min**2,diff_max**2]))
    diff_min_lim = limits*-1
    diff_max_lim = limits

    # Plot the aircraft data.
    plt0 = ax[0].scatter(lon_data,alt_data,c=a_avg,marker='X',s=100,cmap=plt.cm.Spectral_r,vmin=min,vmax=max)
    ax[0].set_ylabel('Altitude / m',fontsize=40,labelpad=20)
    ax[0].tick_params(labelsize=30)
    ax[0].grid()
    ax[0].set_title('Aircraft',fontsize=40)
    cb0 = plt.colorbar(plt0,ax=ax[0],orientation='vertical',pad=0.1,extend='both')
    cb0.set_label(label,fontsize=30,labelpad=10)
    cb0.ax.tick_params(labelsize=20)

    # Plot the model data.
    plt1 = ax[1].scatter(lon_data,alt_data,c=m_avg,marker='X',s=100,cmap=plt.cm.Spectral_r,vmin=min,vmax=max)
    ax[1].set_xlabel('Longitude / degrees east',fontsize=40,labelpad=20)
    ax[1].tick_params(labelsize=30)
    ax[1].grid()
    ax[1].set_title('Model',fontsize=40)
    cb1 = plt.colorbar(plt1,ax=ax[1],orientation='vertical',pad=0.1,extend='both')
    cb1.set_label(label,fontsize=30,labelpad=10)
    cb1.ax.tick_params(labelsize=20)

    # Plot the difference data.
    plt2 = ax[2].scatter(lon_data,alt_data,c=diff_data,marker='X',s=100,cmap=plt.cm.RdBu_r,vmin=diff_min_lim,vmax=diff_max_lim)
    ax[2].tick_params(labelsize=30)
    ax[2].grid()
    ax[2].set_title('Difference',fontsize=40)
    cb2 = plt.colorbar(plt2,ax=ax[2],orientation='vertical',pad=0.1,extend='both')
    cb2.set_label(label+' (Model - Observation)',fontsize=30,labelpad=10)
    cb2.ax.tick_params(labelsize=20)

    # Save the figure.
    plt.savefig(plotdir+key+'_Longitude_Curtain_Plot.png')
    plt.close()

def statistics(df,plotdir,key,label,resample_time,avg_method,min_method,max_method) :

    """
    Calculate model bias and plot a timeseries of bias.
    """

    # Define the figure.
    fig,ax = setup_figure()

    # Define the data.
    time_data     = pd.to_datetime(df.index)
    aircraft_data = df['Aircraft'][:].tolist()
    model_data    = df['Model'][:].tolist()

    # Calculate the bias.
    bias = [x-y for x,y in zip(model_data,aircraft_data)]
    mean_bias = str("{:.2f}".format(np.nanmean(bias)))

    # Resample the data.
    new_df = pd.DataFrame()
    new_df['Bias'] = pd.Series(bias,index=time_data)
    if avg_method == 'mean' :
        avg_df = new_df.resample(resample_time).mean()
    if avg_method == 'median' :
        avg_df = new_df.resample(resample_time).median()
    min_df = new_df.resample(resample_time).quantile(min_method)
    max_df = new_df.resample(resample_time).quantile(max_method)

    # Define the data to plot.
    time_data = avg_df.index
    avg_data  = avg_df['Bias'][:].tolist()
    min_data  = min_df['Bias'][:].tolist()
    max_data  = max_df['Bias'][:].tolist()

    # Plot the data.
    plt.plot(time_data,avg_data,lw=5,c='cadetblue')
    plt.fill_between(time_data,min_data,max_data,fc='cadetblue',ec=None,alpha=0.5)

    # Set the axes labels.
    plt.xlabel('Time / UTC',fontsize=50,labelpad=10)
    plt.ylabel('Model Bias '+label,fontsize=50,labelpad=10)

    # Set the axes ticks.
    time_ticks,time_labels = calculate_time_markers(time_data)
    plt.xticks(time_ticks)
    ax.set_xticklabels(time_labels)
    ax.tick_params(labelsize=30)

    # Add the mean bias as a title.
    plt.title('Mean Bias = '+mean_bias,fontsize=40,y=1.05)

    # Save the figure.
    plt.savefig(plotdir+key+'_Timeseries_Of_Bias.png')
    plt.close()

if __name__ == '__main__' :

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # This section needs changing to the desired settings.
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Set the flight number.
    # Current valid options are M270, M296 and M302.
    flight_number = 'M296'

    # Extract the flight information from the dictionary.
    flight_dict = flight_dictionary()
    flight_date = flight_dict[flight_number]['flight_date']
    start_time  = flight_dict[flight_number]['start_time']
    end_time    = flight_dict[flight_number]['end_time']

    # Define the time interval information.
    # Set the time_filter_flag to True in order to cut out the chosen time range.
    # Set the time_filter_flag to False in order to use the whole flight.
    # The start_time and end_time are defined in the flight dictionary.
    # They can be overwritten with different values here if required.
    time_filter_flag = False
    start_time = 'HH:MM:SS'
    end_time   = 'HH:MM:SS'

    # Define the data directory.
    # This doesn't need changing unless you move the data to a different location.
    modeldir = '../Data_Files/Model/'+flight_number+'/'
    obsdir   = '../Data_Files/Aircraft/'+flight_number+'/'

    # Define the data files.
    # This doesn't need changing unless you rename the files.
    model_track_file    = flight_number + '_' + flight_date + '_Model_Track_Data.csv'
    model_column_file   = flight_number + '_' + flight_date + '_Model_Column_Data.nc'
    aircraft_track_file = flight_number + '_' + flight_date + '_Aircraft_Track_Data.csv'

    # Define the plot directory.
    # This doesn't need changing unless you want to save the plots to a different location.
    plotdir = '../Plots/'+flight_number+'/'
    if not os.path.exists(plotdir) :
        os.makedirs(plotdir)

    # Define the resampling time and averaging methods.
    # The avg_method can be either 'mean' or 'median'.
    # The min_method and max_method refer to the quantile of the data.
    # For the minimum and maximum select 0 and 1, for the interquartile range, select 0.25 and 0.75.
    avg_method = 'mean'
    resample_time = '60S'
    min_method = 0
    max_method = 1

    # Define the bin sizes for altitude (in metres), latitude (in degrees north) and longitude (in degrees east).
    alt_bin = 50
    lat_bin = 0.1
    lon_bin = 0.1

    # Define the colours to be used for the datasets.
    # a_colour = aircraft colour
    # m_colour = model colour
    a_colour = 'darkslategrey'
    m_colour = 'indianred'

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # No need to change things below here
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Read the data.
    aircraft_df = read_data(obsdir,aircraft_track_file)
    model_df    = read_data(modeldir,model_track_file)

    # Read and loop over the species to analyse.
    species_dict = species_dictionary()
    for key in species_dict.keys() :
        code,label,column_key,unit_conv = species_dict[key]['code'],species_dict[key]['label'],species_dict[key]['column_key'],species_dict[key]['unit_conv']

        # Combine the aircraft and model data.
        df = combine_data(aircraft_df,model_df,code,time_filter_flag,start_time,end_time)

        # Resample the data.
        resample_df = resample_data(df,resample_time,avg_method,min_method,max_method)
        wind_df = resample_wind_data(df,resample_time)

        # Read the cross section data.
        column_time,column_alt,column_data = read_cross_section(modeldir,model_column_file,column_key,unit_conv,time_filter_flag,start_time,end_time,flight_date)

        # Plot the data.
        #plot_timeseries(resample_df,plotdir,key,label,a_colour,m_colour)
        #plot_altitude_profile(df,resample_df,plotdir,key,label,a_colour,m_colour,alt_bin,avg_method,min_method,max_method)
        #plot_latitude_profile(df,resample_df,plotdir,key,label,a_colour,m_colour,lat_bin,avg_method,min_method,max_method)
        #plot_longitude_profile(df,resample_df,plotdir,key,label,a_colour,m_colour,lon_bin,avg_method,min_method,max_method)
        #plot_correlations(resample_df,plotdir,key,label)
        plot_maps(resample_df,wind_df,plotdir,key,label)
        #plot_cross_sections(resample_df,plotdir,key,label,column_time,column_alt,column_data)
        #plot_curtain(resample_df,plotdir,key,label)
        #statistics(df,plotdir,key,label,resample_time,avg_method,min_method,max_method)
