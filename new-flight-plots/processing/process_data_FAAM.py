from cmath import nan
import os
import iris
import iris.cube
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from datetime import datetime
from iris.time import PartialDateTime

def process_aircraft_data(obsdir,aircraft_nc_file,aircraft,flight_number,flight_date,start_time,aircraft_csv_file) :

    """
    Read the observational data from the aircraft NetCDF data file and reformat
    into a dataframe with data filtered to include only data flagged as good data.
    Save the subset data as a CSV file.
    """

    def add_variables_to_dataframe(datakey) :
        """
        Loop through the variables in the datakey,
        adding them to the dataframe.
        """
        for variable in datakey:
            data = f.variables[variable[0]][:]
            time = datetime_data
            if variable[0]+'_flag' in f.variables:
                flag = f.variables[variable[0]+'_flag'][:]
                filtered_data = [data[x] for x in range(len(data)) if flag[x] == 0.0]
                time = [datetime_data[x] for x in range(len(datetime_data)) if flag[x] == 0.0]
                if variable[0] == 'AIMMS_temperature': #convert celsius to kelvin
                    data = [x+variable[1] for x in filtered_data]
                else:
                    data = [x*variable[1] for x in filtered_data]
            df[variable[2]] = pd.Series(data,index=time)

    # Define the aircraft observational file name.
    obs_file = aircraft_nc_file+'.nc'

    # Read the aircraft observational data file.
    f = Dataset(obsdir+flight_number+'/'+obs_file,mode='r',format='NETCDF4')

    # Create a blank data frame to hold the extracted aircraft data.
    df = pd.DataFrame()

    # Process the time data and convert the units to seconds since 1970.
    if aircraft == 'MOASA':
        time_data = f.variables['time'][:]
    elif aircraft == 'FAAM':
        time_data = f.variables['Time'][:]
    epoch = datetime(1970,1,1)
    current_date = datetime(int(flight_date[0:4]),int(flight_date[4:6]),int(flight_date[6:8]),
                            int(start_time[0:2]),int(start_time[2:4]),int(start_time[4:6]))
    timedelta = (current_date - epoch).total_seconds()
    time_data = [x+timedelta for x in time_data]
    datetime_data = [pd.to_datetime(x,unit='s') for x in time_data]
    df['Time / seconds since 1970-01-01 00:00:00 UTC'] = pd.Series(time_data,index=datetime_data)

    if aircraft == 'MOASA':

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

        # 2D array specifying conversion: 
        # ["variable name", conversion factor (1 if no conversion needed), "CSV column name"]
        datakey = [
            ['AIMMS_wind_speed' , 1, 'Wind Speed / m s-1'],
            ['AIMMS_pressure' , 1, 'Air Pressure / hPa'],
            ['AIMMS_temperature' , 273.15, 'Air Temperature / K'],
            ['AIMMS_RH' , 1, 'Relative Humidity / %'],
            ['NO2_concentration' , 1, 'NO2 / ppb'],
            ['NO2_concentration', 1.913, 'NO2 / ug m-3'],
            ['OZONE_ppb' , 1, 'O3 / ppb'],
            ['OZONE_ppb' , 1.996, 'O3 / ug m-3'],
            ['SO2_ppb' , 1, 'SO2 / ppb'],
            ['SO2_ppb' , 2.661, 'SO2 / ug m-3'],
            ['POPS_pm25' , 1, 'PM2.5 / ug m-3'],]

        add_variables_to_dataframe(datakey)

        # Filter the whole data frame to remove anywhere that does not have valid coordinate data.
        # It does not make sense to have any meteorological or chemical data if we can't associate
        # it with a location, even if that data is flagged as 'good data'.
        filtered_df = df[df['Latitude / degrees north Flag'] == 0.0]
        filtered_df = filtered_df[filtered_df['Longitude / degrees east Flag'] == 0.0]
        filtered_df = filtered_df[filtered_df['Altitude / m Flag'] == 0.0]

    elif aircraft == 'FAAM':
        # Process the coordinate data.
        lat_data = f.variables['LAT_GIN'][:]
        lon_data = f.variables['LON_GIN'][:]
        alt_data = f.variables['ALT_GIN'][:]        
        df['Latitude / degrees north'] = pd.Series(lat_data,index=datetime_data)       
        df['Longitude / degrees east'] = pd.Series(lon_data,index=datetime_data)       
        df['Altitude / m'] = pd.Series(alt_data,index=datetime_data)

        # 2D array specifying conversion: 
        # ["variable name", conversion factor (1 if no conversion needed), "CSV column name"]
        datakey = [
            ['O3_TECO', 1, 'O3 / ppb'],
            ['CO_AERO', 1, 'CO / ppb'],
            ['PM2.5', 1, 'PM2.5 / ug m-3']] 

        add_variables_to_dataframe(datakey)

        # Filter to remove data which have no location information.
        filtered_df = df[np.isfinite(df['Latitude / degrees north'])]
        filtered_df = filtered_df[np.isfinite(filtered_df['Longitude / degrees east'])]
        filtered_df = filtered_df[np.isfinite(filtered_df['Altitude / m'])]

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

def add_altitude_coord(cube, model_top, z_string):

    cube = cube.copy()
    coord_names = [coord.name() for coord in cube.dim_coords]
    # Get eta values
    eta_strings = cube.coord('z').points
    eta = np.array([float(string.replace("Z = ", "").replace(z_string,"")) for string in eta_strings])

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

    return cube

def process_gridded_model_data(modeldir,flight_number,model,suite,flight_date,gridded_file,model_top,z_string) :

    if model == 'NAME':
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
                pm25_cube = add_altitude_coord(pm25_cube, model_top, z_string)
                total_cubeList.append(pm25_cube)

        # Merge all of the cubes on the time dimension.
        merged_cube = total_cubeList.merge()

    elif model == 'AQUM':
        """
        Read the AQUM model output data into iris.
        The model data is saved in PP file format.
        Each file contains 1 hour of data at 5 minute time intervals (12 time steps per file).
        There is 1 file for meteorological data, 1 file for gasesous data and 1 file for particulate matter data.
        Read all of the hourly data files for the flight day and save as a single file.
        """

        # Load the different PP files containing model output.
        met_cubes = iris.load(modeldir+flight_number+'/prodf*')
        gas_cubes = iris.load(modeldir+flight_number+'/prodg*')
        pm_cubes  = iris.load(modeldir+flight_number+'/prodh*')

        # Combine the meteorology, gas and particulate matter data.
        merged_cube = met_cubes + gas_cubes + pm_cubes

    # Save the combined model data as a NetCDF file.
    iris.save(merged_cube,modeldir+flight_number+suite+'/'+gridded_file)

def filter_gridded_model_data(modeldir,obsdir,flight_number,model,suite,flight_date,gridded_file,filtered_gridded_file,aircraft_csv_file) :

    """
    Filter the gridded model data file down based on flight area and time
    in order to reduce the file size being handled for the column and track extraction.
    """

    # Read the aircraft coordinate data from the CSV file.
    aircraft_times,aircraft_lats,aircraft_lons,aircraft_alts = read_aircraft_data(obsdir,flight_number,flight_date,aircraft_csv_file)

    #Time filtering will work differently for NAME vs AQUM:
    #"The partial datetime functionality doesn't work with time coordinates that have bounds. 
    #By using cell.point you are explicitly telling iris to ignore bounds and just use the cell centre point for comparison."
    #So have just done it manually for now.
    
    if model == 'AQUM':
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
    
    # Latitude and longitude filtering - - - - - - - - - - - - - - - - - - - - -

    # Define the minimum and maximum aircraft latitude.
    aircraft_min_lat = np.nanmin(aircraft_lats)
    aircraft_max_lat = np.nanmax(aircraft_lats)

    # Define the minimum and maximum aircraft longitude.
    aircraft_min_lon = np.nanmin(aircraft_lons)
    aircraft_max_lon = np.nanmax(aircraft_lons)

    # For AQUM, rotate thecoordinates onto a rotated pole system.
    # Then define the minimum lat/lon as 1 degree south/west of the minimum aircraft position and
    # the maximum lat/lon as 1 degree north/east of the maximum aircraft position.
    if model == 'AQUM':
        rot_min_lon,rot_min_lat = iris.analysis.cartography.rotate_pole(np.array(aircraft_min_lon),
                                                                        np.array(aircraft_min_lat),
                                                                        177.5,37.5)
        rot_min_lon = rot_min_lon + 360

        rot_max_lon,rot_max_lat = iris.analysis.cartography.rotate_pole(np.array(aircraft_max_lon),
                                                                        np.array(aircraft_max_lat),
                                                                        177.5,37.5)
        rot_max_lon = rot_max_lon + 360

        min_lat = rot_min_lat[0] -0.5
        max_lat = rot_max_lat[0] +0.5

        min_lon = rot_min_lon[0] -0.5
        max_lon = rot_max_lon[0] +0.5

    elif model == 'NAME':
        min_lat = aircraft_min_lat -0.5
        max_lat = aircraft_max_lat +0.5

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

def process_column_model_data(modeldir,obsdir,flight_number,model,suite,flight_date,column_file,model_cubes,constraints,aircraft_csv_file) :

    """
    Use the aircraft time, latitude and longitude to interpolate the model data for the whole column.
    """

    # Read the aircraft coordinate data.
    aircraft_times,aircraft_lats,aircraft_lons,aircraft_alts = read_aircraft_data(obsdir,flight_number,flight_date,aircraft_csv_file)

    if model == 'AQUM':
        # Rotate the aircraft coordinates onto a rotated grid to match the model data.
        aircraft_rot_lons,aircraft_rot_lats = iris.analysis.cartography.rotate_pole(\
                                        np.array(aircraft_lons),np.array(aircraft_lats),\
                                        177.5,37.5)
        aircraft_rot_lons = [x+360 for x in aircraft_rot_lons]

    # Convert the aircraft time from seconds to hours to match the model data.
    aircraft_times = [x/3600 for x in aircraft_times]
  
    if len(constraints) > 0:
        constraintList = []
        for constraint in constraints:
            constraintList.append(iris.Constraint(name=constraint))
        process_cubes = model_cubes.extract_cubes(constraintList)
    else:
        process_cubes = model_cubes

    # Use the aircraft coordinates to interpolate the model data.
    count = 0
    for cube in process_cubes :
        # Create a temporary list to add cube data to and loop over each aircraft coordinate to interpolate the data.
        temp_cubes = []
        for x in range(len(aircraft_times)) :
            # Define the criteria for interpolation.
            if model == 'AQUM':
                sample_point = [('time',aircraft_times[x]),
                                ('grid_latitude',aircraft_rot_lats[x]),
                                ('grid_longitude',aircraft_rot_lons[x])]

            elif model == 'NAME':           
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
    sample_cube = track_cubes[0]

    # Process the time data.
    time_data = [x*3600 for x in sample_cube.coord('time').points.tolist()]
    datetime_data = [pd.to_datetime(x,unit='s') for x in time_data]
    df['Time / seconds since 1970-01-01 00:00:00 UTC'] = pd.Series(time_data,index=datetime_data)
    
    if model == 'AQUM':
        # Process the rotated coordinate data.
        lat_rot_data = sample_cube.coord('grid_latitude').points.tolist()
        lon_rot_data = sample_cube.coord('grid_longitude').points.tolist()
        lon_data,lat_data = iris.analysis.cartography.unrotate_pole(np.array(lon_rot_data),np.array(lat_rot_data),177.5,37.5)
        
        # Process the wind data.
        xwind_data = track_cubes.extract(iris.Constraint(name='x_wind'))[0].data[:].tolist()
        ywind_data = track_cubes.extract(iris.Constraint(name='y_wind'))[0].data[:].tolist()
        windspeed_data = [np.sqrt(x**2+y**2) for x,y in zip(xwind_data,ywind_data)]
        df['U Wind / m s-1'] = pd.Series(xwind_data,index=datetime_data)
        df['V Wind / m s-1'] = pd.Series(ywind_data,index=datetime_data)
        df['Wind Speed / m s-1'] = pd.Series(windspeed_data,index=datetime_data)

        # 2D array specifying conversion: 
        # ["cube name", conversion factor (1 if no conversion needed), "CSV column name" (matching aircraft data file)]
        datakey = [
        ['atmosphere_boundary_layer_thickness', 1, 'Boundary Layer Thickness / m'],
        ['air_pressure', 0.01, 'Air Pressure / hPa'],
        ['air_temperature', 1, 'Air Temperature / K'],
        ['specific_humidity', 1, 'Specific Humidity / kg kg-1'],
        ['surface_air_pressure', 0.01, 'Surface Air Pressure / hPa'],
        ['surface_temperature', 1, 'Surface Air Temperature / K'],
        ['mass_fraction_of_nitrogen_monoxide_in_air', 1e9, 'NO / ppb'],
        ['mass_fraction_of_nitrogen_monoxide_in_air', 1.248e9, 'NO / ug m-3'],
        ['mass_fraction_of_nitrogen_dioxide_in_air', 1e9, 'NO2 / ppb'],
        ['mass_fraction_of_nitrogen_dioxide_in_air', 1.913e9, 'NO2 / ug m-3'],
        ['mass_fraction_of_ozone_in_air', 1e9, 'O3 / ppb'],
        ['mass_fraction_of_ozone_in_air', 1.996e9, 'O3 / ug m-3'],
        ['mass_fraction_of_sulfur_dioxide_expressed_as_sulfur_in_air', 1e9, 'SO2 / ppb'],
        ['mass_fraction_of_sulfur_dioxide_expressed_as_sulfur_in_air', 2.661e9, 'SO2 / ug m-3'],
        ['mass_fraction_of_carbon_monoxide_in_air', 1e9, 'CO / ppb'],
        ['mass_fraction_of_carbon_monoxide_in_air', 1.165e9, 'CO / ug m-3'],
        ['mass_concentration_of_pm2p5_dry_aerosol_in_air', 1, 'PM2.5 / ug m-3'],
        ['mass_concentration_of_pm10_dry_aerosol_in_air', 1, 'PM10 / ug m-3']]

    elif model == 'NAME':
        # Process the coordinate data.
        lat_data = sample_cube.coord('latitude').points.tolist()
        lon_data = sample_cube.coord('longitude').points.tolist()

        # 2D array specifying conversion: 
        # ["cube name", conversion factor (1 if no conversion needed), "CSV column name" (matching aircraft data file)]
        datakey = [
            ['TOTAL_PM25_CONCENTRATION', 1, 'PM2.5 / ug m-3']]

    # Continue to process the coordinate data.
    df['Latitude / degrees north'] = pd.Series(lat_data,index=datetime_data)
    df['Longitude / degrees east'] = pd.Series(lon_data,index=datetime_data)    
    alt_data = sample_cube.coord('level_height').points.tolist()
    df['Altitude / m'] = pd.Series(alt_data,index=datetime_data)

    # Process the rest of the data from the datakeys.
    for cube in datakey:
        data = track_cubes.extract(iris.Constraint(name=cube[0]))[0].data.tolist()
        conversion = cube[1]
        converted_data = [x*conversion for x in data]
        df[cube[2]] = pd.Series(converted_data,index=datetime_data)

    # Save the CSV file.
    df.to_csv(modeldir+flight_number+suite+'/'+track_csv_file)

if __name__ == '__main__' :

    # Define the flight information and data directory. Something like:
    # model            = 'AQUM'
    # aircraft         = 'MOASA'
    # flight_number    = 'M270'
    # suite            = ''
    # flight_date      = '20200915'
    # aircraft_nc_file = 'clean_air_moasa_data_20200915_M270_v0'
    # OR
    # model            = 'NAME'
    # aircraft         = 'FAAM'
    # flight_number    = 'C110'
    # suite            = '/mi-bd327'
    # flight_date      = '20180629'
    # aircraft_nc_file = 'core_faam_20180629_v004_r0_c110_1hz_with_PM25'

    model            = 'AQUM'
    aircraft         = 'MOASA'
    flight_number    = 'M270'
    suite            = ''
    flight_date      = '20200915'
    aircraft_nc_file = 'clean_air_moasa_data_20200915_M270_v0' #initial netCDF name
    modeldir         = '../Data_Files/Model/'
    obsdir           = '../Data_Files/Aircraft/'

    # As processing column data can be intensive, you may want to only process cubes you're interested in. 
    constraints      = [] # List of cube names, or empty list

    if aircraft == 'FAAM': 
        start_time = '071205' #required as FAAM stores time values from zero.
    elif aircraft == 'MOASA':
        start_time = '000000'

    if model == 'NAME':
        model_top = 80000.000 # metres. model_top may change, depending on model resolution.
        z_string = " UM4km_Mk3 ZCoord" #UMG_Mk5 Zcoord or UM4km_Mk3 ZCoord, defined in model data.
    elif model == 'AQUM':
        model_top = None
        z_string = None

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
        process_aircraft_data(obsdir,aircraft_nc_file,aircraft,flight_number,flight_date,start_time,aircraft_csv_file)

    # Check whether the combined model gridded data exists and if not create the file.
    if not gridded_file in os.listdir(modeldir+flight_number+suite+'/') :
        print('Creating combined gridded model data for flight',flight_number)
        process_gridded_model_data(modeldir,flight_number,model,suite,flight_date,gridded_file,model_top,z_string) 

    # Create a second file which is filtered based on the aircraft coordinates to minimise data file size.
    if not filtered_gridded_file in os.listdir(modeldir+flight_number+suite+'/') :
        print('Creating filtered gridded model data for flight',flight_number)
        filter_gridded_model_data(modeldir,obsdir,flight_number,model,suite,flight_date,gridded_file,filtered_gridded_file,aircraft_csv_file)
    model_cubes = read_gridded_model_data(modeldir,flight_number,suite,filtered_gridded_file)

    # Check whether the column interpolated model data exists and if not create the file.
    if not column_file in os.listdir(modeldir+flight_number+suite+'/') :
        print('Creating column interpolated model data for flight',flight_number)
        process_column_model_data(modeldir,obsdir,flight_number,model,suite,flight_date,column_file,model_cubes,constraints,aircraft_csv_file)
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