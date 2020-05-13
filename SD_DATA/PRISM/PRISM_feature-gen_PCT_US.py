# general tools
import sys
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np
import netCDF4 as nc

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/')
import data_utils as du
from namelist import * 

VAR_list = ['PCT']; #thres = 2.5e-1 

# datetime list
base = datetime(2015, 1, 1, 0)
date_list = [base + timedelta(days=x) for x in range(365+366+365+365+1)]

# datetime info
base = datetime(2015, 1, 1, 0)
N_days = 365 + 366 + 365 + 365 + 365 # 2015-2020 (period ending)
date_list = [base + timedelta(days=x) for x in range(N_days)]

# number of days in each month (for mm/month --> mm/day)
mon_days_365 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
mon_days_366 = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# import geographical variables
with h5py.File(PRISM_dir+'PRISM_regrid_2015_2020.hdf', 'r') as hdf_io:
    # etopo
    etopo_4km = hdf_io['etopo_4km'][...]
    etopo_regrid = hdf_io['etopo_regrid'][...]
    # lon/lat
    lon_4km = hdf_io['lon_4km'][...]
    lat_4km = hdf_io['lat_4km'][...]
    lon_025 = hdf_io['lon_025'][...]
    lat_025 = hdf_io['lat_025'][...]
    # land_mask
    land_mask = hdf_io['land_mask'][...]
# non negative elevation correction
etopo_4km[etopo_4km<0] = 0
etopo_regrid[etopo_regrid<0] = 0

# dictionary (tuple)
dict_save = {}

# hdf5 labels
label_save = []

# loop over variables
for VAR in VAR_list:
    print('===== Process {} ===== '.format(VAR))
    
    # PCT
    with h5py.File(PRISM_dir+'PRISM_regrid_2015_2020.hdf', 'r') as hdf_io:
        PRISM_P  = hdf_io['{}_4km'.format(VAR)][...]
        REGRID_P = hdf_io['{}_REGRID'.format(VAR)][...]

    # CLIM
    with h5py.File(PRISM_dir+'PRISM_regrid_clim.hdf', 'r') as hdf_io:
        CLIM_4km = hdf_io['{}_4km'.format(VAR)][...]
        CLIM_REGRID = hdf_io['{}_REGRID'.format(VAR)][...]
    
    CLIM_4km_duplicate = np.zeros(PRISM_P.shape)
    CLIM_REGRID_duplicate = np.zeros(PRISM_P.shape)

    # duplicate climatology to NRT
    for i, date in enumerate(date_list):
        ind = date.month-1
        if date.year == 2016:
            # leap year handling (not the best way but works)
            CLIM_4km_duplicate[i, ...] = CLIM_4km[ind, ...]/mon_days_366[ind]
            CLIM_REGRID_duplicate[i, ...] = CLIM_REGRID[ind, ...]/mon_days_366[ind]
        else:
            CLIM_4km_duplicate[i, ...] = CLIM_4km[ind, ...]/mon_days_365[ind]
            CLIM_REGRID_duplicate[i, ...] = CLIM_REGRID[ind, ...]/mon_days_365[ind]
            
    # other feature operations #
    REGRID_P[REGRID_P<0] = 0
    CLIM_REGRID_duplicate[CLIM_REGRID_duplicate<0] = 0
    
    PRISM_P = du.log_trans(PRISM_P)
    REGRID_P = du.log_trans(REGRID_P)
    CLIM_4km_duplicate = du.log_trans(CLIM_4km_duplicate)
    CLIM_REGRID_duplicate = du.log_trans(CLIM_REGRID_duplicate)
    
    REGRID_P[REGRID_P<thres] = 0
    # ------------------------ #

    # save data
    dict_save['{}_4km'.format(VAR)] = PRISM_P
    dict_save['{}_REGRID'.format(VAR)] = REGRID_P
    dict_save['{}_CLIM_4km'.format(VAR)] = CLIM_4km_duplicate
    dict_save['{}_CLIM_REGRID'.format(VAR)] = CLIM_REGRID_duplicate
    # collecting label
    label_save.append('{}_4km'.format(VAR))
    label_save.append('{}_REGRID'.format(VAR))
    label_save.append('{}_CLIM_4km'.format(VAR))
    label_save.append('{}_CLIM_REGRID'.format(VAR))
    
    # dictionary to tuple
    tuple_etopo = (etopo_4km, etopo_regrid)
    tuple_grids = (lon_025, lat_025, lon_4km, lat_4km, land_mask)
    # mark labels
    label_etopo = ['etopo_4km', 'etopo_regrid']
    label_grids = ['lon_025', 'lat_025', 'lon_4km', 'lat_4km', 'land_mask']

    # save hdf
    tuple_save = tuple(dict_save.values()) + tuple_etopo + tuple_grids
    label_all = label_save + label_etopo + label_grids
    du.save_hdf5(tuple_save, label_all, out_dir=PRISM_dir, filename='PRISM_{}_features_2015_2020.hdf'.format(VAR))

