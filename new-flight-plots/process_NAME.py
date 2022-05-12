import iris
import iris.quickplot as qplt
import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs
import matplotlib.dates as mdates
import iris.pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

model_top = 80000.000 # m
etacst = 0.2194822

savedir = "/data/users/twilson/cap-notebooks/new flight plots/Data_Files/Model/C110/"

models = {
    "12km" : "mi-bd328",
    "12km_high" : "mi-bd591",
    "2p2km" : "mi-bd327",
    "2p2km_high" : "mi-bd592"
    
}

sites = {
    'Salford_Eccles' : (53.484810, -2.334139),
    'Manchester_Picadilly' : (53.481520, -2.237881),
    'Wigan_Centre' : (53.549140, -2.638139),
}

def add_altitude_coord(cube):
    
    cube = cube.copy()

    # Get eta values
    eta_strings = cube.coord('z').points
    eta = np.array([float(string.replace("Z = ", "").replace(" UMG_Mk5 ZCoord","")) for string in eta_strings])

    # Replace values in z aux coord with level heights
    cube.coord("z").points = eta * model_top
    cube.coord("z").units = "m"
    iris.util.promote_aux_coord_to_dim_coord(cube, "z")

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
    
    return cube
file = savedir+"20180629T0000Z_Fields_grideta_C1_T338_201806290100.txt"
cubes = iris.load(file)
#print(cubes)
cube = cubes.extract_cube("PM25_CONCENTRATION")
print(cube)
coord_names = [coord.name() for coord in cube.coords()]
df = pd.read_table(file, delimiter=',', skiprows=38, index_col=0, usecols=[2,3])
print(df)
#print(coord_names)
#df = iris.pandas.as_data_frame(cube)
#print(df)

"""for model in models:

    cubes = iris.load(f"/scratch/bdrummon/mass_retrievals/name_3d/{models[model]}/2018062[56789]T0000Z_Fields_grideta_C1_T???_201806????00.txt")
    orogcube = iris.load_cube(f"/scratch/bdrummon/mass_retrievals/name_3d/{models[model]}/*gridorog*")
    
    for site in sites:
        
        site_lat = sites[site][0]
        site_lon = sites[site][1]
    
        cube = cubes.extract_cube("PM25_CONCENTRATION")
        cube = add_altitude_coord(cube)

        cube_lats = cube.coord('latitude').points
        cube_lons = cube.coord('longitude').points

        latid = np.argmin(abs(site_lat - cube_lats))
        lonid = np.argmin(abs(site_lon - cube_lons))

        cube = cube.extract(iris.Constraint(latitude=cube_lats[latid]))
        cube = cube.extract(iris.Constraint(longitude=cube_lons[lonid]))

        # Units
        cube = cube * 1e6
        cube.units = 'ug/m3'


for day in [25,26,27,28,29]:
    
    cubes = iris.load(f"{datadir}/201806{day}T0000Z_Fields_grideta_C1_T???_201806{day}1[34]00.txt")
    
    cube = cubes.extract("CO_CONCENTRATION", strict=True)
    cube = add_altitude_coord(cube)

    altitude = cube.coord('altitude').points

    # calculate altitude depths
    eta_boundary = []
    with open("etagrid_boundary.txt", "r") as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            eta_boundary.append(float(row[0]))
    eta_boundary = np.array(eta_boundary)
    z_boundary = eta_boundary * model_top
    b = (1 - eta_boundary/etacst)**2
    altitude_boundary = z_boundary[:,np.newaxis, np.newaxis] + b[:,np.newaxis, np.newaxis]*orogcube.data
    daltitude = altitude_boundary[1:,:,:] - altitude_boundary[0:-1,:,:]

    column_total = cube.collapsed('z', iris.analysis.SUM, weights=daltitude)
    # Convert to mol
    column_total = column_total/28.01
    column_total.units = 'mol m-2'"""

