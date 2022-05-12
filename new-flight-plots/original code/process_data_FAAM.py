from cmath import nan
import os
import iris
import iris.cube
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from datetime import datetime
from iris.time import PartialDateTime

def process_aircraft_data(obsdir,flight_number,flight_date,aircraft_csv_file) :

    """
    Read the observational data from the aircraft NetCDF data file and reformat
    into a dataframe with data filtered to include only data flagged as good data.
    Save the subset data as a CSV file.
    """

    # Define the aircraft observational file name.
    obs_file = 'clean_air_moasa_data_'+flight_date+'_'+flight_number+'_v0.nc'

    # Read the aircraft observational data file.
    f = Dataset(obsdir+flight_number+'/'+obs_file,mode='r',format='NETCDF4')

    # Create a blank data frame to hold the extracted aircraft data.
    df = pd.DataFrame()

    # Process the time data and convert the units to seconds since 1970.
    time_data = f.variables['time'][:]
    epoch = datetime(1970,1,1)
    current_date = datetime(int(flight_date[0:4]),int(flight_date[4:6]),int(flight_date[6:8]))
    timedelta = (current_date - epoch).total_seconds()
    time_data = [x+timedelta for x in time_data]
    datetime_data = [pd.to_datetime(x,unit='s') for x in time_data]
    df['Time / seconds since 1970-01-01 00:00:00 UTC'] = pd.Series(time_data,index=datetime_data)

    # Process the coordinate data.
    # Read the latitude, longitude and altitude data and flags and add the data to the data frame.
    lat_data = f.variables['AIMMS_latitude'][:]
    lon_data = f.variables['AIMMS_longitude'][:]
    alt_data = f.variables['AIMMS_altitude'][:]
    df['Latitude / degrees north'] = pd.Series(lat_data,index=datetime_data)
    df['Longitude / degrees east'] = pd.Series(lon_data,index=datetime_data)
    df['Altitude / m'] = pd.Series(alt_data,index=datetime_data)
    lat_flag = f.variables['AIMMS_latitude_flag'][:]
    lon_flag = f.variables['AIMMS_longitude_flag'][:]
    alt_flag = f.variables['AIMMS_altitude_flag'][:]
    df['Latitude / degrees north Flag'] = pd.Series(lat_flag,index=datetime_data)
    df['Longitude / degrees east Flag'] = pd.Series(lon_flag,index=datetime_data)
    df['Altitude / m Flag'] = pd.Series(alt_flag,index=datetime_data)

    # Process the wind data.
    # Read the wind speed data and flag and remove any data points which are not flagged as good data.
    windspeed_data = f.variables['AIMMS_wind_speed'][:]
    windspeed_flag = f.variables['AIMMS_wind_speed_flag'][:]
    windspeed_data = [windspeed_data[x] for x in range(len(windspeed_data)) if windspeed_flag[x] == 0.0]
    windspeed_time = [datetime_data[x] for x in range(len(datetime_data)) if windspeed_flag[x] == 0.0]
    df['Wind Speed / m s-1'] = pd.Series(windspeed_data,index=windspeed_time)

    # Process the air pressure data.
    # Read the air pressure data and flag and remove any data points which are not flagged as good data.
    airpressure_data = f.variables['AIMMS_pressure'][:]
    airpressure_flag = f.variables['AIMMS_pressure_flag'][:]
    airpressure_data = [airpressure_data[x] for x in range(len(airpressure_data)) if airpressure_flag[x] == 0.0]
    airpressure_time = [datetime_data[x] for x in range(len(datetime_data)) if airpressure_flag[x] == 0.0]
    df['Air Pressure / hPa'] = pd.Series(airpressure_data,index=airpressure_time)

    # Process the air temperature data.
    # Read the air temperature data and flag, remove any data points which are not flagged as good data and convert
    # the temperature from degrees celcius to kelvin.
    airtemp_data = f.variables['AIMMS_temperature'][:]
    degc_to_k_conversion = 273.15
    airtemp_data = [x+degc_to_k_conversion for x in airtemp_data]
    airtemp_flag = f.variables['AIMMS_temperature_flag'][:]
    airtemp_data = [airtemp_data[x] for x in range(len(airtemp_data)) if airtemp_flag[x] == 0.0]
    airtemp_time = [datetime_data[x] for x in range(len(datetime_data)) if airtemp_flag[x] == 0.0]
    df['Air Temperature / K'] = pd.Series(airtemp_data,index=airtemp_time)

    # Process the relative humidity data.
    # Read the relative humidity data and flag and remove any data points which are not flagged as good data.
    relhumidity_data = f.variables['AIMMS_RH'][:]
    relhumidity_flag = f.variables['AIMMS_RH_flag'][:]
    relhumidity_data = [relhumidity_data[x] for x in range(len(relhumidity_data)) if relhumidity_flag[x] == 0.0]
    relhumidity_time = [datetime_data[x] for x in range(len(datetime_data)) if relhumidity_flag[x] == 0.0]
    df['Relative Humidity / %'] = pd.Series(relhumidity_data,index=relhumidity_time)

    # Process the NO2 data.
    # Read the concentration data and flag, remove any data points which are not flagged as good data,
    # convert the units to micrograms per metre cubed and include both units in the data frame.
    no2_data = f.variables['NO2_concentration'][:]
    no2_flag = f.variables['NO2_concentration_flag'][:]
    no2_data = [no2_data[x] for x in range(len(no2_data)) if no2_flag[x] == 0.0]
    no2_time = [datetime_data[x] for x in range(len(datetime_data)) if no2_flag[x] == 0.0]
    df['NO2 / ppb'] = pd.Series(no2_data,index=no2_time)
    no2_ppb_ugm3_conversion = 1.913
    no2_data = [x*no2_ppb_ugm3_conversion for x in no2_data]
    df['NO2 / ug m-3'] = pd.Series(no2_data,index=no2_time)

    # Process the O3 data.
    # Read the concentration data and flag, remove any data points which are not flagged as good data,
    # convert the units to micrograms per metre cubed and include both units in the data frame.
    o3_data = f.variables['OZONE_ppb'][:]
    o3_flag = f.variables['OZONE_ppb_flag'][:]
    o3_data = [o3_data[x] for x in range(len(o3_data)) if o3_flag[x] == 0.0]
    o3_time = [datetime_data[x] for x in range(len(datetime_data)) if o3_flag[x] == 0.0]
    df['O3 / ppb'] = pd.Series(o3_data,index=o3_time)
    o3_ppb_ugm3_conversion  = 1.996
    o3_data = [x*o3_ppb_ugm3_conversion for x in o3_data]
    df['O3 / ug m-3'] = pd.Series(o3_data,index=o3_time)

    # Process the SO2 data.
    # Read the concentration data and flag, remove any data points which are not flagged as good data,
    # convert the units to micrograms per metre cubed and include both units in the data frame.
    so2_data = f.variables['SO2_ppb'][:]
    so2_flag = f.variables['SO2_ppb_flag'][:]
    so2_data = [so2_data[x] for x in range(len(so2_data)) if so2_flag[x] == 0.0]
    so2_time = [datetime_data[x] for x in range(len(datetime_data)) if so2_flag[x] == 0.0]
    df['SO2 / ppb'] = pd.Series(so2_data,index=so2_time)
    so2_ppb_ugm3_conversion = 2.661
    so2_data = [x*so2_ppb_ugm3_conversion for x in so2_data]
    df['SO2 / ug m-3'] = pd.Series(so2_data,index=so2_time)

    # Process the PM2.5 data.
    # Read the concentration data and flag and remove any data points which are not flagged as good data.
    pm2p5_data = f.variables['POPS_pm25'][:]
    df['PM2.5 / ug m-3'] = pd.Series(pm2p5_data,index=datetime_data)

    # Filter the whole data frame to remove anywhere that does not have valid coordinate data.
    # It does not make sense to have any meteorological or chemical data if we can't associate
    # it with a location, even if that data is flagged as 'good data'.
    filtered_df = df[df['Latitude / degrees north Flag'] == 0.0]
    filtered_df = filtered_df[filtered_df['Longitude / degrees east Flag'] == 0.0]
    filtered_df = filtered_df[filtered_df['Altitude / m Flag'] == 0.0]

    # Save the CSV file.
    filtered_df.to_csv(obsdir+flight_number+'/'+aircraft_csv_file)

def read_aircraft_data(obsdir,flight_number,flight_date,aircraft_csv_file) :

    """
    Read the processed aircraft observational data CSV file and read the coordinates of the flight track.
    """

    # Read the data file.
    df = pd.read_csv(obsdir+flight_number+'/'+aircraft_csv_file,index_col=0)

    # Define the aircraft coordinate data.
    aircraft_times = df['Time / seconds since 1970-01-01 00:00:00 UTC'][:].tolist()
    aircraft_lats  = df['Latitude / degrees north'][:].tolist()
    aircraft_lons  = df['Longitude / degrees east'][:].tolist()
    aircraft_alts  = df['Altitude / m'][:].tolist()

    # Return this data.
    return aircraft_times, aircraft_lats, aircraft_lons, aircraft_alts

def add_altitude_coord(cube):

    # These constants may change, depending on model resolution.
    model_top = 80000.000 # m
    etacst = 0.2194822
    #orogcube = iris.load_cube(modeldir+flight_number+suite+'/orog/'+"20180615T0000Z_Fields_gridorog_C1_T1_201806150000.txt")

    cube = cube.copy()
    coord_names = [coord.name() for coord in cube.dim_coords]
    # Get eta values
    eta_strings = cube.coord('z').points
    eta = np.array([float(string.replace("Z = ", "").replace(" UMG_Mk5 ZCoord","")) for string in eta_strings])

    # Check if last eta value is smaller
    if eta[-1] < eta[-2]:
        etanew = eta.copy()
        etanew[0] = eta[-1]
        etanew[1:] = eta[0:-1]

        datanew = cube.data.copy()
        if 'time' in coord_names:
            datanew[:,0,:,:] = cube.data[:,-1,:,:]
            datanew[:,1:,:,:] = cube.data[:,0:-1,:,:]
        else:
            datanew[0,:,:] = cube.data[-1,:,:]
            datanew[1:,:,:] = cube.data[0:-1,:,:]
        eta = etanew
        cube.data = datanew

    # Replace values in z aux coord with level heights

    cube.coord("z").points = eta * model_top
    cube.coord("z").units = "m"
    cube.coord("z").rename("level_height")
    iris.util.promote_aux_coord_to_dim_coord(cube, "level_height")
    """
    # Create altitude 
    b = (1 - eta/etacst)**2
    altitude =  eta[:, np.newaxis, np.newaxis] * model_top + b[:, np.newaxis, np.newaxis] * orogcube.data

    # Create altitude aux coord
    altcoord = iris.coords.AuxCoord(altitude, long_name='altitude', units='m')
    
    # Add aux coord, checking on data dimensions
    coord_names = [coord.name() for coord in cube.dim_coords]
    if 'time' in coord_names:
        data_dims = (1,2,3)
    else:
        data_dims = (0,1,2)
    cube.add_aux_coord(altcoord, data_dims=data_dims)
    """
    return cube

def process_gridded_model_data(modeldir,flight_number,suite,flight_date,gridded_file) :

    """
    Read the NAME model output data into iris.
    The model data is saved in txt file format.
    Each file contains 1 hour of data, averaged.
    Read all of the hourly data files for the flight day, add the altitude dimension coord, and save as a single file.
    """

    total_cubeList = iris.cube.CubeList([])
    PM25_constituents = ['SULPHATE_CONCENTRATION', 'NH42SO4_CONCENTRATION', 'NH4NO3_CONCENTRATION']
    path = modeldir+flight_number+suite
    for file in os.listdir(path):
        name, ext = os.path.splitext(file)
        if ext == '.txt':
            # Load the cube list from each file.
            hour_cubeList = iris.load(modeldir+flight_number+suite+'/'+file)
            #extract pm25, loop over cube list and add constraints,renamecube TOTAL_PM25_CONCENTRATION
            pm25_cube = hour_cubeList.extract_cube(iris.Constraint(name="PM25_CONCENTRATION"))

            for cube in hour_cubeList:
                if cube.name() in PM25_constituents:
                    pm25_cube += cube

            pm25_cube.rename('TOTAL_PM25_CONCENTRATION')
            pm25_cube.convert_units('ug / m^3')
            pm25_cube = add_altitude_coord(pm25_cube)
            total_cubeList.append(pm25_cube)

    # Merge all of the cubes on the time dimension.
    merged_cube = total_cubeList.merge()

    # Save the combined model data as a NetCDF file.
    iris.save(merged_cube,modeldir+flight_number+suite+'/'+gridded_file)

def filter_gridded_model_data(modeldir,obsdir,flight_number,suite,flight_date,gridded_file,filtered_gridded_file,aircraft_csv_file) :

    """
    Filter the gridded model data file down based on flight area and time
    in order to reduce the file size being handled for the column and track extraction.
    """

    # Read the aircraft coordinate data from the CSV file.
    aircraft_times,aircraft_lats,aircraft_lons,aircraft_alts = read_aircraft_data(obsdir,flight_number,flight_date,aircraft_csv_file)

    #Time filtering will work differently for NAME vs AQUM:
    #"The partial datetime functionality doesn't work with time coordinates that have bounds (I don't know why not). 
    #By using cell.point you are explicitly telling iris to ignore bounds and just use the cell centre point for comparison."
    #So have just done it manually for now.
    """
    # Time filtering - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Define the first and last aircraft data point.
    start_hour   = int((pd.to_datetime(aircraft_times[0],unit='s')).hour)
    start_minute = int((pd.to_datetime(aircraft_times[0],unit='s')).minute)
    end_hour     = int((pd.to_datetime(aircraft_times[-1],unit='s')).hour)
    end_minute   = int((pd.to_datetime(aircraft_times[-1],unit='s')).minute)

    # Define the start time of the filtering as 10 minutes before the
    # first aircraft data point and the end time as 10 minutes after
    # the last data point.
    start_time = PartialDateTime(year=int(flight_date[0:4]),month=int(flight_date[4:6]),
                                 day=int(flight_date[6:8]),hour=start_hour,minute=start_minute-10)                                  
    end_time = PartialDateTime(year=int(flight_date[0:4]),month=int(flight_date[4:6]),
                               day=int(flight_date[6:8]),hour=end_hour,minute=end_minute+10)

    # Define the date time constraint for loading the data.
    time_filter = iris.Constraint(time=lambda cell: start_time < cell < end_time)
    """
    # Latitude and longitude filtering - - - - - - - - - - - - - - - - - - - - -

    # Define the minimum and maximum aircraft latitude.
    aircraft_min_lat = np.nanmin(aircraft_lats)
    aircraft_max_lat = np.nanmax(aircraft_lats)

    # Define the minimum and maximum aircraft longitude.
    aircraft_min_lon = np.nanmin(aircraft_lons)
    aircraft_max_lon = np.nanmax(aircraft_lons)

    
    # Rotate the minimum coordiantes onto a rotated pole system.
    rot_min_lon,rot_min_lat = iris.analysis.cartography.rotate_pole(np.array(aircraft_min_lon),
                                                                    np.array(aircraft_min_lat),
                                                                    177.5,37.5)
    rot_min_lon = rot_min_lon + 360

    # Rotate the maximum coordinates onto a rotated pole system.
    rot_max_lon,rot_max_lat = iris.analysis.cartography.rotate_pole(np.array(aircraft_max_lon),
                                                                    np.array(aircraft_max_lat),
                                                                    177.5,37.5)
    rot_max_lon = rot_max_lon + 360

    # Define the minimum latitude as 1 degree south of the minimum aircraft position and
    # the maximum latitude as 1 degree north of the maximum aircraft position.
    min_lat = aircraft_min_lat -0.5
    max_lat = aircraft_max_lat +0.5

    # Define the minimum longitude as 1 degree west of the minimum aircraft position and
    # the maximum longitude as 1 degree east of the maximum aircraft position.
    min_lon = aircraft_min_lon -0.5
    max_lon = aircraft_max_lon +0.5

    # Define the latitude constraint for loading the data.
    lat_filter = iris.Constraint(latitude=lambda cell: min_lat < cell < max_lat)

    # Define the longitude constraint for loading the data.
    lon_filter = iris.Constraint(longitude=lambda cell: min_lon < cell < max_lon)

    # Read and filter the data to a new file - - - - - - - - - - - - - - - - - -

    # Read the filtered data.
    model_cube = iris.load(modeldir+flight_number+suite+'/'+gridded_file,lat_filter & lon_filter)

    # Save the combined model data as a NetCDF file.
    iris.save(model_cube,modeldir+flight_number+suite+'/'+filtered_gridded_file)

def read_gridded_model_data(modeldir,flight_number,suite,filtered_gridded_file) :

    """
    Read the subset of the AQUM gridded model data into iris from the NetCDF file.
    """

    gridded_cubes = iris.load(modeldir+flight_number+suite+'/'+filtered_gridded_file)

    return gridded_cubes

def process_column_model_data(modeldir,obsdir,flight_number,suite,flight_date,column_file,model_cubes,aircraft_csv_file) :

    """
    Use the aircraft time, latitude and longitude to interpolate the model data for the whole column.
    """

    # Read the aircraft coordinate data.
    aircraft_times,aircraft_lats,aircraft_lons,aircraft_alts = read_aircraft_data(obsdir,flight_number,flight_date,aircraft_csv_file)

    # Convert the aircraft time from seconds to hours to match the model data.
    aircraft_times = [x/3600 for x in aircraft_times]

    # Use the aircraft coordinates to interpolate the model data.
    count = 0
    constraint = iris.Constraint(name="TOTAL_PM25_CONCENTRATION")
    cube = model_cubes.extract_cube(constraint)

    # Create a temporary list to add cube data to and loop over each aircraft coordinate to interpolate the data.
    temp_cubes = []
    for x in range(len(aircraft_times)) :

        if not np.isnan(aircraft_lats[x]) or not np.isnan(aircraft_lons[x]):
        # Define the criteria for interpolation.
            sample_point = [('time',aircraft_times[x]),
                            ('latitude',aircraft_lats[x]),
                            ('longitude',aircraft_lons[x])]
            # Interpolate the data point and add it to the temporary list
            interpolated_cube = cube.interpolate(sample_point,iris.analysis.Linear())
            temp_cubes.append(interpolated_cube)

    # Create a Cube List from the temporary list, merge the cubes and add them to the overall Cube List.
    temp_cubes = iris.cube.CubeList(temp_cubes)
    merged_cube = temp_cubes.merge()

    if count == 0 :
        column_cubes = merged_cube
    else :
        column_cubes = column_cubes + merged_cube
    count += 1

    # Save the interpolated model data as a NetCDF file.
    column_cubes = iris.cube.CubeList(column_cubes)
    iris.save(column_cubes,modeldir+flight_number+suite+'/'+column_file)

def read_column_model_data(modeldir,flight_number,suite,column_file) :

    """
    Read the AQUM model data for the columns along the aircraft track into iris from the NetCDF file.
    """

    column_cubes = iris.load(modeldir+flight_number+suite+'/'+column_file)

    return column_cubes

def process_track_model_data(modeldir,obsdir,flight_number,suite,flight_date,track_file,column_cubes,aircraft_csv_file) :

    """
    Use the aircraft time, latitude, longitude and altitude to interpolate the model data along the flight track.
    """

    # Read the aircraft coordinate data.
    aircraft_times,aircraft_lats,aircraft_lons,aircraft_alts = read_aircraft_data(obsdir,flight_number,flight_date,aircraft_csv_file)

    # Convert the aircraft time from seconds to hours to match the model data.
    aircraft_times = [x/3600 for x in aircraft_times]

    # Loop over each cube and use the aircraft coordinates to interpolate the model data.
    count = 0
    for cube in column_cubes :

        # Check whether this data is surface level or model level data.
        if cube.ndim == 2 :

            # Create a temporary list of cube and loop over each point.
            temp_cubes = []
            for x in range(len(aircraft_times)) :

                # Define the sample point.
                sample_point = [('time',aircraft_times[x]),('level_height',aircraft_alts[x])]

                # Interpolate the data.
                interpolated_cube= cube.interpolate(sample_point,iris.analysis.Linear())
                temp_cubes.append(interpolated_cube)

            # Create a Cube List from the temporary list, merge and add them to the overall Cube List.
            temp_cubes = iris.cube.CubeList(temp_cubes)
            merged_cube = temp_cubes.merge()
            if count == 0 :
                track_cubes = merged_cube
            else :
                track_cubes = track_cubes + merged_cube
            count += 1

        # Add surface level data to the cube list without processing.
        elif cube.ndim == 1 :

            cube = iris.cube.CubeList([cube])
            if count == 0 :
                track_cubes = cube
            else :
                track_cubes = track_cubes + cube
            count += 1

    # Save the interpolated model data as a NetCDF file.
    track_cubes = iris.cube.CubeList(track_cubes)
    iris.save(track_cubes,modeldir+flight_number+suite+'/'+track_file)

def read_track_model_data(modeldir,flight_number,suite,track_file) :

    """
    Read the AQUM model data along the aircraft track into iris from the NetCDF file.
    """

    track_cubes = iris.load(modeldir+flight_number+suite+'/'+track_file)

    return track_cubes

def convert_track_data_to_csv(modeldir,flight_number,suite,track_cubes,track_csv_file) :
    
    """
    Convert the interpolated model track data into CSV format including unit conversions to match the aircraft data file.
    """

    # Create a blank data frame.
    df = pd.DataFrame()

    # Define a sample cube to extract the coordinate information from.
    sample_cube = track_cubes.extract(iris.Constraint(name='TOTAL_PM25_CONCENTRATION'))[0]

    # Process the time data.
    time_data = [x*3600 for x in sample_cube.coord('time').points.tolist()]
    datetime_data = [pd.to_datetime(x,unit='s') for x in time_data]
    df['Time / seconds since 1970-01-01 00:00:00 UTC'] = pd.Series(time_data,index=datetime_data)

    # Process the coordinate data.
    lat_data = sample_cube.coord('latitude').points.tolist()
    lon_data = sample_cube.coord('longitude').points.tolist()
    df['Latitude / degrees north'] = pd.Series(lat_data,index=datetime_data)
    df['Longitude / degrees east'] = pd.Series(lon_data,index=datetime_data)    
    alt_data = sample_cube.coord('level_height').points.tolist()
    df['Altitude / m'] = pd.Series(alt_data,index=datetime_data)

    """
    # Process the wind data.
    xwind_data = track_cubes.extract(iris.Constraint(name='x_wind'))[0].data[:].tolist()
    ywind_data = track_cubes.extract(iris.Constraint(name='y_wind'))[0].data[:].tolist()
    windspeed_data = [np.sqrt(x**2+y**2) for x,y in zip(xwind_data,ywind_data)]
    df['U Wind / m s-1'] = pd.Series(xwind_data,index=datetime_data)
    df['V Wind / m s-1'] = pd.Series(ywind_data,index=datetime_data)
    df['Wind Speed / m s-1'] = pd.Series(windspeed_data,index=datetime_data)

    # Process the boundary layer height data.
    blheight_data = track_cubes.extract(iris.Constraint(name='atmosphere_boundary_layer_thickness'))[0].data.tolist()
    df['Boundary Layer Thickness / m'] = pd.Series(blheight_data,index=datetime_data)

    # Process the air pressure data.
    airpressure_data = track_cubes.extract(iris.Constraint(name='air_pressure'))[0].data[:].tolist()
    pressure_pa_to_hpa_conversion = 0.01
    airpressure_data = [x*pressure_pa_to_hpa_conversion for x in airpressure_data]
    df['Air Pressure / hPa'] = pd.Series(airpressure_data,index=datetime_data)

    # Process the air temperature data.
    airtemp_data = track_cubes.extract(iris.Constraint(name='air_temperature'))[0].data[:].tolist()
    df['Air Temperature / K'] = pd.Series(airtemp_data,index=datetime_data)

    # Process the specific humidity data.
    spechumidity_data = track_cubes.extract(iris.Constraint(name='specific_humidity'))[0].data[:].tolist()
    df['Specific Humidity / kg kg-1'] = pd.Series(spechumidity_data,index=datetime_data)

    # Process the surface air pressure data.
    surfaceairpressure_data = track_cubes.extract(iris.Constraint(name='surface_air_pressure'))[0].data.tolist()
    pressure_pa_to_hpa_conversion = 0.01
    surfaceairpressure_data = [x*pressure_pa_to_hpa_conversion for x in surfaceairpressure_data]
    df['Surface Air Pressure / hPa'] = pd.Series(surfaceairpressure_data,index=datetime_data)

    # Process the surface air temperature data.
    surfaceairtemp_data = track_cubes.extract(iris.Constraint(name='surface_temperature'))[0].data.tolist()
    df['Surface Air Temperature / K'] = pd.Series(surfaceairtemp_data,index=datetime_data)
    

    # Process the NO data.
    no_data = track_cubes.extract(iris.Constraint(name='mass_fraction_of_nitrogen_monoxide_in_air'))[0].data[:].tolist()
    kgkg_ppb_conversion  = 1e9
    no_ppb_data = [x*kgkg_ppb_conversion for x in no_data]
    df['NO / ppb'] = pd.Series(no_ppb_data,index=datetime_data)
    no_kgkg_ugm3_conversion = 1.248e9
    no_ugm3_data = [x*no_kgkg_ugm3_conversion for x in no_data]
    df['NO / ug m-3'] = pd.Series(no_ugm3_data,index=datetime_data)
"""
    # Process the PM2.5 data.
    pm2p5_data = track_cubes.extract(iris.Constraint(name='TOTAL_PM25_CONCENTRATION'))[0].data[:].tolist()
    df['PM2.5 / ug m-3'] = pd.Series(pm2p5_data,index=datetime_data)

    # Save the CSV file.
    df.to_csv(modeldir+flight_number+suite+'/'+track_csv_file)

if __name__ == '__main__' :

    # Define the flight information and data directory.
    # Current available options are:
    flight_number = 'C110'
    suite         = '/mi-bd591'
    flight_date   = '20180629'
    modeldir      = '../Data_Files/Model/'
    obsdir        = '../Data_Files/Aircraft/'

    # Define the model file names.
    aircraft_csv_file     = flight_number + '_' + flight_date + '_Aircraft_Track_Data.csv'
    gridded_file          = flight_number + '_' + flight_date + '_Model_Gridded_Data.nc'
    filtered_gridded_file = flight_number + '_' + flight_date + '_Model_Gridded_Data_Filtered.nc'
    column_file           = flight_number + '_' + flight_date + '_Model_Column_Data.nc'
    track_file            = flight_number + '_' + flight_date + '_Model_Track_Data.nc'
    track_csv_file        = flight_number + '_' + flight_date + '_Model_Track_Data.csv'

    # Check whether the processed aircraft data exists and if not create the file.
    if not aircraft_csv_file in os.listdir(obsdir+flight_number+'/') :
        print('Processing aircraft data for flight',flight_number)
        process_aircraft_data(obsdir,flight_number,flight_date,aircraft_csv_file)
        #    creates ../Data_Files/Aircraft/C110/C110_20180629_Aircraft_Track_Data

    # Check whether the combined model gridded data exists and if not create the file.
    if not gridded_file in os.listdir(modeldir+flight_number+suite+'/') :
        print('Creating combined gridded model data for flight',flight_number)
        process_gridded_model_data(modeldir,flight_number,suite,flight_date,gridded_file) 
        # creates Model_Gridded_Data.nc

    # Create a second file which is filtered based on the aircraft coordinates to minimise data file size.
    if not filtered_gridded_file in os.listdir(modeldir+flight_number+suite+'/') :
        print('Creating filtered gridded model data for flight',flight_number)
        filter_gridded_model_data(modeldir,obsdir,flight_number,suite,flight_date,gridded_file,filtered_gridded_file,aircraft_csv_file)
    model_cubes = read_gridded_model_data(modeldir,flight_number,suite,filtered_gridded_file)

    # Check whether the column interpolated model data exists and if not create the file.
    if not column_file in os.listdir(modeldir+flight_number+suite+'/') :
        print('Creating column interpolated model data for flight',flight_number)
        process_column_model_data(modeldir,obsdir,flight_number,suite,flight_date,column_file,model_cubes,aircraft_csv_file)
    column_cubes = read_column_model_data(modeldir,flight_number,suite,column_file)

    # Check whether the track interpolated model data exists and if not create the file.
    if not track_file in os.listdir(modeldir+flight_number+suite+'/') :
        print('Creating track interpolated model data for flight',flight_number)
        process_track_model_data(modeldir,obsdir,flight_number,suite,flight_date,track_file,column_cubes,aircraft_csv_file)
    track_cubes = read_track_model_data(modeldir,flight_number,suite,track_file)
 
    # Process the track interpolated data further.
    if not track_csv_file in os.listdir(modeldir+flight_number+suite+'/') :
        print('Processing track interpolated model data for flight',flight_number)
        convert_track_data_to_csv(modeldir,flight_number,suite,track_cubes,track_csv_file)