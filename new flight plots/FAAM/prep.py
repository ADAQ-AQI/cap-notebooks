import pandas as pd

def species_dictionary() :

    """
    Create a dictionary of species information for creating the figures.
    This information includes the species codes as used in the data files,
    labels to use for figure axes.
    """

    species_dict = {'O3'    : {'code':  'O3 / ppb',
                            'label': 'O$_3$ / ppb'},
                    'CO'    : {'code':  'CO / ppb',
                            'label': 'CO / ppb'},
                    'PM2.5' : {'code':  'PM2.5 / ug m-3',
                            'label': 'PM$_2$$_.$$_5$ / \u03BCg m$^-$$^3$'}}

    return species_dict

def read_data(datadir,datafile) :

    """
    Read the data file into a data frame.
    """

    df = pd.read_csv(datadir+datafile,index_col=0)
    df.index = pd.to_datetime(df.index)

    return df

flight_number = 'C110'
flight_date   = '20180629'
aircraft_dir = './Data_Files/Aircraft/'
aircraft_csv_file = 'C110_20180629_Aircraft_Track_Data.csv'

data = read_data(aircraft_dir, aircraft_csv_file)

plotdir = './Plots/'