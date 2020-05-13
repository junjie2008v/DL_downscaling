# general tools
import sys
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/utils/')
import data_utils as du
import verif_utils as vu
import pipeline_utils as pu
from namelist import * 

clim = False # contains climatology input or not.

# import geo info
with h5py.File(PRISM_dir+'PRISM_TMAX_features_2015_2020.hdf', 'r') as hdf_io:
    etopo_regrid = hdf_io['etopo_regrid'][...]
    etopo_4km = hdf_io['etopo_4km'][...]
    # lat/lon
    land_mask = hdf_io['land_mask'][...]
    lon_4km = hdf_io['lon_4km'][...]
    lat_4km = hdf_io['lat_4km'][...]

# macros
seasons = ['djf', 'mam', 'jja', 'son']

# dt list for all
N_all = 365 + 366 + 365 + 365 + 365
all_list = [datetime(2015, 1, 1, 0) + timedelta(days=x) for x in range(N_all)]

# dt list for train and valid (test)
N_train = 365 + 366 + 365 # 2015-2018 (period ending)
N_valid = 365 # 2018-2019 (period ending)
train_list = [datetime(2015, 1, 1, 0) + timedelta(days=x) for x in range(N_train)]
valid_list = [datetime(2018, 1, 1, 0) + timedelta(days=x) for x in range(N_valid)]

# pick ind
ind_train = du.dt_match(all_list, train_list)
ind_valid = du.dt_match(all_list, valid_list)
# inds
ind_train_sea  = du.season_ind_sep(train_list, key_format='{}')
ind_valid_sea  = du.season_ind_sep(valid_list, key_format='{}')
# shape
grid_shape = land_mask.shape
N_no_clim = 3 # total number of features in the "no clim" case
N_with_clim = 4 # total number of features in the "with clim" case

# loop over variables
VARS = ['TMAX', 'TMIN']
for var in VARS:
    print('===== {} ====='.format(var))
    C_no_clim = np.zeros(grid_shape+(4*N_no_clim,))
    I_no_clim = np.zeros(grid_shape+(4*1,))
    OUT_no_clim = np.zeros((N_valid,)+grid_shape)
    
    if clim:   
        C_with_clim = np.zeros(grid_shape+(4*N_with_clim,))
        I_with_clim = np.zeros(grid_shape+(4*1,))
        OUT_with_clim = np.zeros((N_valid,)+grid_shape)
    
    with h5py.File(PRISM_dir+'PRISM_{}_features_2015_2020.hdf'.format(var), 'r') as hdf_io:
        T_4km = hdf_io['{}_4km'.format(var)][...]
        T_REGRID = hdf_io['{}_REGRID'.format(var)][...]
        T_CLIM_4km = hdf_io['{}_CLIM_4km'.format(var)][...]
    
    # separate by seasons
    for i, sea in enumerate(seasons):
        print('season: {}'.format(sea))
        # inds
        ind_train_temp = ind_train_sea[sea]
        ind_valid_temp = ind_valid_sea[sea]
        
        Y = T_4km[ind_train, ...][ind_train_temp, ...]
        X_2d = (etopo_regrid, etopo_4km, )
        
        print('\t no climatology input')
        X_3d_train = (T_REGRID[ind_train, ...][ind_train_temp, ...],)
        X_3d_valid = (T_REGRID[ind_valid, ...][ind_valid_temp, ...],)
        I, C, OUT = vu.baseline_estimator(X_3d_train, X_2d, Y, X_3d_valid, X_2d, land_mask)
        OUT_no_clim[ind_valid_temp, ...] = OUT
        C_no_clim[..., N_no_clim*i:N_no_clim*(i+1)] = C
        I_no_clim[..., i] = I
        
        if clim:
            print('\t with climatology input')
            X_3d_train = (T_REGRID[ind_train, ...][ind_train_temp, ...], T_CLIM_4km[ind_train, ...][ind_train_temp, ...])
            X_3d_valid = (T_REGRID[ind_valid, ...][ind_valid_temp, ...], T_CLIM_4km[ind_valid, ...][ind_valid_temp, ...])
            I, C, OUT = vu.baseline_estimator(X_3d_train, X_2d, Y, X_3d_valid, X_2d, land_mask)
            OUT_with_clim[ind_valid_temp, ...] = OUT
            C_with_clim[..., N_with_clim*i:N_with_clim*(i+1)] = C
            I_with_clim[..., i] = I
            
    if clim:
        # save as hdf
        tuple_save = (lon_4km, lat_4km, C_no_clim, I_no_clim, OUT_no_clim, C_with_clim, I_with_clim, OUT_with_clim)
        label_save = ['lon_4km', 'lat_4km', 'C_no_clim', 'I_no_clim', 'OUT_no_clim', 'C_with_clim', 'I_with_clim', 'OUT_with_clim']
        du.save_hdf5(tuple_save, label_save, out_dir=save_dir, filename='BASELINE_PRISM_{}_2018_2020.hdf'.format(var))
    else:
        # save as hdf
        tuple_save = (lon_4km, lat_4km, C_no_clim, I_no_clim, OUT_no_clim)
        label_save = ['lon_4km', 'lat_4km', 'C_no_clim', 'I_no_clim', 'OUT_no_clim']
        du.save_hdf5(tuple_save, label_save, out_dir=save_dir, filename='BASELINE_PRISM_{}_2018_2020.hdf'.format(var))
