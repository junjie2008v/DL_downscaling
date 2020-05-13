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
with h5py.File(PRISM_dir+'land_mask_NA.hdf', 'r') as hdf_io:
    lon_4km = hdf_io['lon_4km'][...]
    lat_4km = hdf_io['lat_4km'][...]
    lon_025_clean = hdf_io['lon_025'][...]
    lat_025_clean = hdf_io['lat_025'][...]
#     etopo_regrid_clean = hdf_io['etopo_regrid'][...]
    land_mask = hdf_io['land_mask'][...]

print('Raw netCDF4 process')
# access original lat, lon
with nc.Dataset(ERA_TMEAN_dir + 'ei.oper.an.sfc.regn128sc.2019083118.sha416621.nc', 'r') as nc_io:
    x025 = nc_io['g4_lon_2'][...]-360
    y025 = nc_io['g4_lat_1'][...]
    
y025 = np.flipud(y025) # <--- revert lat sequence
lon_025, lat_025 = np.meshgrid(x025, y025)

print('Preparing ETOPO data')
with nc.Dataset(BACKUP_dir+'ETOPO1_Ice_g_gmt4.grd') as nc_obj:
    etopo_x = nc_obj.variables['x'][2000:] # subsetting north america
    etopo_y = nc_obj.variables['y'][6000:]
    etopo_z = nc_obj.variables['z'][6000:, 2000:]
etopo_lon, etopo_lat = np.meshgrid(etopo_x, etopo_y)
# interp.
etopo_025 = du.interp2d_wraper(etopo_lon, etopo_lat, etopo_z, lon_025, lat_025, method=interp_method)
etopo_regrid = du.interp2d_wraper(lon_025, lat_025, etopo_025, lon_4km, lat_4km, method=interp_method)


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

# allocation
freq = 4
grid_shape = lon_025.shape
temp_var = np.empty((freq,)+grid_shape)
era_var = np.empty((days_ref,)+grid_shape)
era_interp = np.empty((days_ref,)+lon_4km.shape)
era_025 = np.empty((days_ref,)+lon_025_clean.shape)

for VAR in VARS:
    print('Process {}'.format(VAR))
    count = 0
    f = eval(method[VAR])
    for i in range(days_ref):
        for j in range(freq):
            file_num = freq*i+j
            with nc.Dataset(filenames[VAR][file_num], 'r') as nc_io:
                temp_field = nc_io.variables[nc_keys[VAR]][0, ...]
                temp_field = np.flipud(temp_field)
                temp_var[j, ...] = temp_field
                
        # aggregate from 6-hr to daily
        era_var[i, ...] = f(temp_var, axis=0)

    print('Interpolation')
    for i in range(days_ref):
        if i%200 == 0:
            print('\tday index: {}'.format(i))
        era_025[i, ...] = du.interp2d_wraper(lon_025, lat_025, era_var[i, ...], 
                                             lon_025_clean, lat_025_clean, method=interp_method)
        temp_interp = du.interp2d_wraper(lon_025, lat_025, era_var[i, ...], lon_4km, lat_4km, method=interp_method)
        # land mask applied
        temp_interp[land_mask] = np.nan
        era_interp[i, ...] = temp_interp
        
    print('Feature engineering')
    if VAR in ['TMAX', 'TMIN', 'TMEAN']:
        print('\tK to C')
        era_025 = era_025 - 273.15
        era_var = era_var - 273.15
        era_interp = era_interp - 273.15

    data_save = (lon_025, lat_025, lon_4km, lat_4km, era_var, era_025, era_interp, etopo_025, land_mask)
    label_save = ['lon_raw', 'lat_raw', 'lon_4km', 'lat_4km', 'era_raw', 'era_025', 'TMEAN_REGRID', 'etopo_raw', 'land_mask']
    du.save_hdf5(data_save, label_save, ERA_dir, 'ERA_{}_features_2015_2020.hdf'.format(VAR))
