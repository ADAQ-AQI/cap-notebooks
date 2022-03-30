import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

flight_number = 'C110'
flight_date   = '20180629'
aircraft_csv_file = '/data/users/ersmith/Clean_Air/Framework/Aircraft_Data_Analysis/FAAM/Data_Files/Aircraft/C110_20180629_Aircraft_Track_Data.csv'

plotdir = '/data/users/ersmith/Clean_Air/Framework/Aircraft_Data_Analysis/FAAM/Plots/'
if not os.path.exists(plotdir) :
    os.makedirs(plotdir)

df = pd.read_csv(aircraft_csv_file,index_col=0)
df.index = pd.to_datetime(df.index)

species_dict = {'O3'    : {'code':  'O3 / ppb',
                           'label': 'O$_3$ / ppb'},
                'CO'    : {'code':  'CO / ppb',
                           'label': 'CO / ppb'},
                'PM2.5' : {'code':  'PM2.5 / ug m-3',
                           'label': 'PM$_2$$_.$$_5$ / \u03BCg m$^-$$^3$'}}

for species in species_dict :
    species_key = species
    species_code = species_dict[species]['code']
    species_label = species_dict[species]['label']

    datetime_data = df.index
    conc_data = df[species_code][:].values

    fig = plt.figure(figsize=(15,15))
    ax  = plt.axes([.2,.15,.75,.7])

    plt.plot(datetime_data,conc_data,lw=5,c='cadetblue',label='Aircraft')

    plt.xlabel('Time / UTC',fontsize=50,labelpad=10)
    plt.ylabel(species_label,fontsize=50,labelpad=10)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%m'))
    ax.tick_params(labelsize=30)

    plt.legend(fontsize=30,markerscale=2)

    plt.savefig(plotdir+species_key+'_Timeseries.png')
    plt.close()
