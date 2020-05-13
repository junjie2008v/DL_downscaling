import sys
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
# import geo information from feature files
with h5py.File(PRISM_dir+'PRISM_TMEAN_features_2015_2020.hdf', 'r') as hdf_io:
    lon_4km = hdf_io['lon_4km'][...]
    lat_4km = hdf_io['lat_4km'][...]
    land_mask = hdf_io['land_mask'][...]
    etopo_4km = hdf_io['etopo_4km'][...]
    etopo_regrid = hdf_io['etopo_regrid'][...]
    TMEAN_4km = hdf_io['TMEAN_4km'][...]

print('Raw netCDF4 process')
# ERA is stored every 12-hour
# Vars:
#     T2: 2T_GDS4_SFC
#     lon: g4_lon_2
#     lat: g4_lat_1
# Time:
#     2015-01-01 to 2019-08-31
# data has been subsetted to NA (matches with land mask)

# access original lat, lon
with nc.Dataset(ERA_TMEAN_dir + 'ei.oper.fc.sfc.regn128sc.2019083112.sha410420.nc', 'r') as nc_io:
    x025 = nc_io['g4_lon_2'][...]-360
    y025 = nc_io['g4_lat_1'][...]
lon_025, lat_025 = np.meshgrid(x025, y025)

# processing keywords
VARS = ['TMEAN']
# nc variable keys
nc_keys = {}
nc_keys['TMEAN'] = '2T_GDS4_SFC' # analysis
# filenames
filenames = {}
filenames['TMEAN'] = sorted(glob(ERA_TMEAN_dir+'*.nc'))
# aggregation method
method = {}
method['TMEAN'] = 'nanmean'

# datetime info
days_ref = 365 + 366 + 365 + 365 + 243 # 2015 - 2019-08-31

# !!! <---- assuming no missing date
# datetime reference (in the case of missing date)
#base = datetime(2015, 1, 1, 0)
#date_list = [base + timedelta(days=x) for x in range(days_ref)]
#print('last day: {}'.format(date_list[-1]))

mon_days_365 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
mon_days_366 = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# allocation
grid_shape = lon_025.shape
temp_var = np.empty((2,)+grid_shape)
era_var = np.empty((days_ref,)+grid_shape)
era_interp = np.empty((days_ref,)+lon_4km.shape)

for VAR in VARS:
    print('Process {}'.format(VAR))
    count = 0
    f = eval(method[VAR])
    for i in range(days_ref):
        for j in range(2):
            file_num = 2*i+j
            with nc.Dataset(filenames[VAR][file_num], 'r') as nc_io:
                temp_var[j, ...] = nc_io.variables[nc_keys[VAR]][0, ...]
                
        # aggregate from 6-hr to daily
        era_var[i, ...] = f(temp_var, axis=0)

    print('Interpolation')
    for i in range(days_ref):
        if i%200 == 0:
            print('\tday index: {}'.format(i))
        temp_interp = du.interp2d_wraper(lon_025, lat_025, era_var[i, ...], lon_4km, lat_4km, method=interp_method)
        # land mask applied
        temp_interp[land_mask] = np.nan
        era_interp[i, ...] = temp_interp
        
    print('Feature engineering')
    if VAR in ['TMAX', 'TMIN', 'TMEAN']:
        print('\tK to C')
        era_var = era_var - 273.15
        era_interp = era_interp - 273.15
    
    # match ERA with 4km PRISM on time
    TMEAN_4km = TMEAN_4km[:days_ref, ...]
    
    data_save = (lon_4km, lat_4km, era_interp, TMEAN_4km, etopo_regrid, land_mask)
    label_save = ['lon_4km', 'lat_4km', '{}_REGRID'.format(VAR), 'TMEAN_4km', 'etopo_regrid', 'land_mask']
    du.save_hdf5(data_save, label_save, ERA_dir, 'ERA_{}_features_US_2015_2020.hdf'.format(VAR))




    



