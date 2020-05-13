import sys
import pickle
from os.path import basename
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np
from numpy import nansum, nanmax, nanmin, nanmean
import netCDF4 as nc

# geo tools
import shapely
from scipy.interpolate import griddata
from cartopy.io.shapereader import Reader

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/')
import data_utils as du
from namelist import * 

print('Import lat/lon and land mask')
with h5py.File(PRISM_dir+'land_mask_NA.hdf', 'r') as hdf_io:
    lon_4km = hdf_io['lon_4km'][...]
    lat_4km = hdf_io['lat_4km'][...]
    lon_025_clean = hdf_io['lon_025'][...]
    lat_025_clean = hdf_io['lat_025'][...]
#     etopo_regrid_clean = hdf_io['etopo_regrid'][...]
    land_mask = hdf_io['land_mask'][...]

    
    

    data_save = (lon_025, lat_025, lon_4km, lat_4km, era_var, era_025, era_interp, etopo_025, land_mask)
    label_save = ['lon_raw', 'lat_raw', 'lon_4km', 'lat_4km', 'era_raw', 'era_025', 'TMEAN_REGRID', 'etopo_raw', 'land_mask']
    du.save_hdf5(data_save, label_save, ERA_dir, 'ERA_{}_features_2015_2020.hdf'.format(VAR))
