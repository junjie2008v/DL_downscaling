
'''
T2 downscaling features with climatology
 - no temperature signal fit
'''

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
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/utils/')
import data_utils as du
import pipeline_utils as pu
from namelist import * 

sine_fit = False # <------- !!!!!

VAR_list = ['TMAX', 'TMIN', 'TMEAN'] # PCT not here because of different feature eng.

# datetime info
N_days = 365 + 366 + 365 + 365 + 365 # 2015-2020 (period ending)
date_list = [datetime(2015, 1, 1, 0) + timedelta(days=x) for x in range(N_days)]

N_train = 365 + 366 + 365 
train_list = [datetime(2015, 1, 1, 0) + timedelta(days=x) for x in range(N_train)]
ind_train = du.dt_match(date_list, train_list)

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

# macros for sine fit
shape_025 = lon_025.shape # 0.25 degree grid shape
shape_4km = lon_4km.shape # 0.25 degree grid shape

if sine_fit:
    # time axis values (2*pi = 1 year)
    # time axis by Julian days (for mean val sine fit)
    t_365 = np.linspace(0, 2*np.pi, 365)
    t_366 = np.linspace(0, 2*np.pi, 366)
    t_all = np.concatenate((t_365, t_366, t_365, t_365, t_365), axis=0)
    t_train = np.concatenate((t_365, t_366, t_365), axis=0)
    # time axis by month (for stddev sine fit)
    days_365 = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    days_366 = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    cumsum_365 = np.cumsum(np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]))
    cumsum_366 = np.cumsum(np.array([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]))
    t_midmon365 = np.zeros(12)
    t_midmon366 = np.zeros(12)
    for i in range(12):
        ceil_365 = np.ceil(0.5*days_365[i])
        ceil_366 = np.ceil(0.5*days_366[i])
        floor_365 = np.floor(0.5*days_365[i])
        floor_366 = np.floor(0.5*days_366[i])
        t_midmon365[i] = 0.5*(t_365[int(cumsum_365[i]+floor_365)] + t_365[int(cumsum_365[i]+ceil_365)])
        t_midmon366[i] = 0.5*(t_366[int(cumsum_366[i]+floor_366)] + t_366[int(cumsum_366[i]+ceil_366)])
    t_midmon = (2/3)*(t_midmon365) + (1/3)*(t_midmon366) # 

# dictionary (tuple)
dict_save = {}

# hdf5 labels
label_save = []

# loop over variables
for VAR in VAR_list:
    print('===== Process {} ===== '.format(VAR))
    
    # T2
    print('NRT PRISM processing')
    with h5py.File(PRISM_dir+'PRISM_regrid_2015_2020.hdf', 'r') as hdf_io:
        RAW_T = hdf_io['{}_025'.format(VAR)][...]
        PRISM_T = hdf_io['{}_4km'.format(VAR)][...]
        REGRID_T = hdf_io['{}_REGRID'.format(VAR)][...]
        
    # CLIM
    print('Duplicating climatology fileds')
    with h5py.File(PRISM_dir+'PRISM_regrid_clim.hdf', 'r') as hdf_io:
        CLIM_4km = hdf_io['{}_4km'.format(VAR)][...]
        CLIM_REGRID = hdf_io['{}_REGRID'.format(VAR)][...]
    
    CLIM_4km_duplicate = np.zeros(PRISM_T.shape)
    CLIM_REGRID_duplicate = np.zeros(PRISM_T.shape)

    # duplicate climatology to NRT
    for i, date in enumerate(date_list):
        ind = date.month-1
        CLIM_4km_duplicate[i, ...] = CLIM_4km[ind, ...]
        CLIM_REGRID_duplicate[i, ...] = CLIM_REGRID[ind, ...]
        
    # ----- other feature operations ----- #
    if sine_fit:
        print('Feature engineering with sine function fit')

        ## grid-point-wise mean value sine fit
        print('\tFit mean')
        MEAN_sinefit = np.zeros((N_days,) + shape_025)
        for i in range(shape_025[0]):
            for j in range(shape_025[1]):
                A, p, c = pu.fit_sin_annual(t_train, RAW_T[ind_train, i, j])
                MEAN_sinefit[:, i, j] = pu.sinfunc_annual(t_all, A, p, c)

        ## grid-point-wise stddev sine fit
        print('\tFit stddev')
        diff_T = RAW_T - MEAN_sinefit
        STD_sinefit = np.zeros((N_days,) + shape_025)
        # temp allocation, repeatly used per grid point, 31*3 for 3 year, 2015-2018
        month_sep = np.zeros([12, 31*3])*np.nan
        for i in range(shape_025[0]):
            for j in range(shape_025[1]):
                count = [0]*12
                temp_grid_point = diff_T[ind_train, i, j]
                for k, date in enumerate(train_list):
                    mon_ind = date.month - 1
                    month_sep[mon_ind, count[mon_ind]] = temp_grid_point[k]
                    count[mon_ind] += 1
                A, p, c = pu.fit_sin_annual(t_midmon, np.nanstd(month_sep, axis=1))
                STD_sinefit[:, i, j] = pu.sinfunc_annual(t_all, A, p, c)
                
        ## interpolate fitted sine to 4-km (faster than interpolate first, and then do the fit)
        STD_4km = np.zeros((N_days,) + shape_4km)
        MEAN_4km = np.zeros((N_days,) + shape_4km)
        for i in range(N_days):
            STD_4km[i, ...] = du.interp2d_wraper(lon_025, lat_025, STD_sinefit[i, ...], lon_4km, lat_4km, method=interp_method)
            MEAN_4km[i, ...] = du.interp2d_wraper(lon_025, lat_025, MEAN_sinefit[i, ...], lon_4km, lat_4km, method=interp_method)

        ## apply to 4km and REGRID
        PRISM_T = (PRISM_T - MEAN_4km)/(STD_4km) # "+1" for numerical stability, optional
        REGRID_T = (REGRID_T - MEAN_4km)/(STD_4km)
        
        dict_save['{}_STD'.format(VAR)] = STD_4km
        dict_save['{}_MEAN'.format(VAR)] = MEAN_4km
        label_save.append('{}_STD'.format(VAR))
        label_save.append('{}_MEAN'.format(VAR))
    # ------------------------ #
    
    # collecting fields
    dict_save['{}_4km'.format(VAR)] = PRISM_T
    dict_save['{}_REGRID'.format(VAR)] = REGRID_T
    dict_save['{}_CLIM_4km'.format(VAR)] = CLIM_4km_duplicate
    dict_save['{}_CLIM_REGRID'.format(VAR)] = CLIM_REGRID_duplicate
    
#     # collecting label
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
    if sine_fit:
        du.save_hdf5(tuple_save, label_all, out_dir=PRISM_dir, filename='PRISM_{}_features_fit_2015_2020.hdf'.format(VAR))
    else:
        du.save_hdf5(tuple_save, label_all, out_dir=PRISM_dir, filename='PRISM_{}_features_2015_2020.hdf'.format(VAR))
