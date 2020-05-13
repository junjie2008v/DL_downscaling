'''
Aggregating, subseting and re-griding NCEP GDAS/FNL data
in the **BC** domain. Land mask and ETOPO1 provided.
'''

import sys
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

# download bounding box setting:
#    lon = [-145, -100]
#    lat = [24, 61]
y_ncep = np.arange(24, 61.25, 0.25)
x_ncep = np.arange(215, 260.25, 0.25)-360
lon_ncep, lat_ncep = np.meshgrid(x_ncep, y_ncep)
# BC domain subset indices identified by hand (single file test)
domain_ind = [20, 128, 96, 149]

# import pre-processed data
CLIM = {}
CLIM_REGRID = {}
with h5py.File(PRISM_dir + 'PRISM_regrid_BC_clim.hdf', 'r') as hdf_io:
    # clim. fileds
    CLIM['PCT'] = hdf_io['PCT_4km'][...]
    CLIM['TMAX'] = hdf_io['TMAX_4km'][...]
    CLIM['TMIN'] = hdf_io['TMIN_4km'][...]
    CLIM['TMEAN'] = hdf_io['TMEAN_4km'][...]
    # regridded clim. files
    CLIM_REGRID['PCT'] = hdf_io['PCT_REGRID'][...]
    CLIM_REGRID['TMAX'] = hdf_io['TMAX_REGRID'][...]
    CLIM_REGRID['TMIN'] = hdf_io['TMIN_REGRID'][...]
    CLIM_REGRID['TMEAN'] = hdf_io['TMEAN_REGRID'][...]
    # BC domain geo info
    lon_4km = hdf_io['lon_4km'][...]
    lat_4km = hdf_io['lat_4km'][...]
    lon_025 = hdf_io['lon_025'][...]
    lat_025 = hdf_io['lat_025'][...]
    land_mask = hdf_io['land_mask'][...]
    etopo_4km = hdf_io['etopo_4km'][...]
    etopo_025 = hdf_io['etopo_025'][...]
    etopo_regrid = hdf_io['etopo_regrid'][...]

# params for preparing clim fields
mon_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
base = datetime(2016, 1, 1, 0)
date_list = [base + timedelta(days=x) for x in range(366+365+364)]
    
# nc variable keys
nc_keys = {}
nc_keys['PCT'] = 'APCP_P8_L1_GLL0_acc' # 6-hr fcst
nc_keys['TMAX'] = 'TMAX_P8_L103_GLL0_max' # 6-hr fcst
nc_keys['TMIN'] = 'TMIN_P8_L103_GLL0_min' # 6-hr fcst
nc_keys['TMEAN'] = 'TMP_P0_L103_GLL0' # analysis
# filenames
filenames = {}
filenames['PCT'] = sorted(glob(NCEP_PCT_dir+'*.nc'))
filenames['TMAX'] = sorted(glob(NCEP_TMAX_dir+'*.nc'))
filenames['TMIN'] = sorted(glob(NCEP_TMIN_dir+'*.nc'))
filenames['TMEAN'] = sorted(glob(NCEP_TMEAN_dir+'*.nc'))
# aggregation method
method = {}
method['PCT'] = 'nansum'
method['TMAX'] = 'nanmax'
method['TMIN'] = 'nanmin'
method['TMEAN'] = 'nanmean'

# grid shapes
grid_shape = lon_ncep.shape
shape_4km = lon_4km.shape

VARS = ['PCT', 'TMAX', 'TMIN', 'TMEAN']

# loop over variables
days_ref = 366 + 365 + 365 + 365 # 2016 - 2020
# datetime reference (in the case of missing date)
base = datetime(2016, 1, 1, 0)
date_list = [base + timedelta(days=x) for x in range(days_ref)]
mon_days_365 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
mon_days_366 = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

for var in VARS:
    print('===== {} ====='.format(var))
    print('Processing raw NCEP files')

    # number of days
    L = len(filenames[var]); days = int(L/4)
    if L % 4 > 0 or days < days_ref:
        print('\tWarning: have missing files')
    
    # aggregation method
    f = eval(method[var])
    
    # allocation
    temp_var = np.empty((4,)+grid_shape)
    var_4km = np.empty((days,)+shape_4km)
    clim_4km = np.empty((days,)+shape_4km)
    clim_regrid = np.empty((days,)+shape_4km)
    ncep_var = np.empty((days,)+grid_shape)
    
    # loop over files
    for i in range(days):
        for j in range(4):
            file_num = 4*i+j
            with nc.Dataset(filenames[var][file_num], 'r') as nc_io:
                # single-time files
                temp_var[j, ...] = np.flipud(nc_io.variables[nc_keys[var]][0, ...]) # flipud on y-axis
        # aggregate from 6-hr to daily
        ncep_var[i, ...] = f(temp_var, axis=0)

    print('BC domain interpolation')
    for i in range(days):
        if i%200 == 0:
            print('\tday index: {}'.format(i))
        temp_interp = du.interp2d_wraper(lon_ncep, lat_ncep, ncep_var[i, ...], lon_4km, lat_4km, method=interp_method)
        # land mask applied
        temp_interp[land_mask] = np.nan
        var_4km[i, ...] = temp_interp
    
    print('Feature engineering')
    if var in ['TMAX', 'TMIN', 'TMEAN']:
        print('\tK to C')
        var_4km = var_4km-273.15
        ncep_var = ncep_var-273.15
        print('Merging climatology fields')
        for i in range(days):
            mon_id = date_list[i].month-1
            clim_4km[i, ...] = CLIM[var][mon_id, ...]
            clim_regrid[i, ...] = CLIM_REGRID[var][mon_id, ...]
    else:
        print('\tPCT log transformation')
        var_4km[var_4km<0] = 0
        ncep_var[ncep_var<0] = 0
        var_4km = du.log_trans(var_4km)
        ncep_var = du.log_trans(ncep_var)
        var_4km[var_4km<thres] = 0
        print('Merging climatology fields')
        for i in range(days):
            mon_id = date_list[i].month-1
            if date_list[i] == 2016:
                clim_4km[i, ...] = CLIM[var][mon_id, ...]/mon_days_366[mon_id]
                clim_regrid[i, ...] = CLIM_REGRID[var][mon_id, ...]/mon_days_366[mon_id]
            else:
                clim_4km[i, ...] = CLIM[var][mon_id, ...]/mon_days_365[mon_id]
                clim_regrid[i, ...] = CLIM_REGRID[var][mon_id, ...]/mon_days_365[mon_id]
        
        clim_4km = du.log_trans(clim_4km)
        clim_regrid = du.log_trans(clim_regrid)
        clim_regrid[clim_regrid<thres] = 0
    
    # save data
    data_save = (lon_4km, lat_4km,
                 lon_ncep[domain_ind[0]:domain_ind[1], domain_ind[2]:domain_ind[3]], 
                 lat_ncep[domain_ind[0]:domain_ind[1], domain_ind[2]:domain_ind[3]], 
                 var_4km, clim_4km, clim_regrid,
                 ncep_var[:, domain_ind[0]:domain_ind[1], domain_ind[2]:domain_ind[3]], 
                 etopo_4km, etopo_regrid, land_mask)
    label_save = ['lon_4km', 'lat_4km', 'lon_ncep', 'lat_ncep', 
                  '{}_REGRID'.format(var), '{}_CLIM_4km'.format(var), '{}_CLIM_REGRID'.format(var),
                  '{}_originals'.format(var), 
                  'etopo_4km', 'etopo_regrid', 'land_mask']
    du.save_hdf5(data_save, label_save, NCEP_dir, 'NCEP_{}_features_BC_2016_2020.hdf'.format(var))