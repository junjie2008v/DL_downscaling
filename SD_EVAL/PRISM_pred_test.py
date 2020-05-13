# general tools
import sys
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np

# ANN tools
from tensorflow import keras

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/utils/')
import data_utils as du
import verif_utils as vu
import pipeline_utils as pu
from namelist import * 

# import geo. info
hdf_io = h5py.File(PRISM_dir + 'PRISM_TMAX_features_2015_2020.hdf', 'r')
land_mask = hdf_io['land_mask'][...]
etopo_4km = hdf_io['etopo_4km'][...]
etopo_regrid = hdf_io['etopo_regrid'][...]
lon_4km = hdf_io['lon_4km'][...]
lat_4km = hdf_io['lat_4km'][...]
hdf_io.close()

grid_shape = land_mask.shape

# defining prediction range
# dt list for all
N_all = 365 + 366 + 365 + 365 + 365
all_list = [datetime(2015, 1, 1, 0) + timedelta(days=x) for x in range(N_all)]
# dt list for pred, 2018-2020
N_pred = 15
pred_list = [datetime(2019, 1, 1, 0) + timedelta(days=x) for x in range(N_pred)]
# indices
ind_pred = du.dt_match(all_list, pred_list)
ind_pred_sea  = du.season_ind_sep(pred_list, key_format='{}')

# macros
# ind_trans = 504 # now in the namelist
model_import_dir = temp_dir
# overlapped tile prediction settings
param = {}
param['gap'] = 8
param['edge'] = 32
param['size'] = 128


# loop over variables and seasons
VARS = ['TMAX', 'TMIN', 'TMEAN']
seasons = ['djf', 'mam', 'jja', 'son']

for var in VARS:
    print('========== {} =========='.format(var))
    
    # allocation
    RESULT_UNET = np.zeros((N_pred,)+grid_shape)
    
    # import 3d (time, lat, lon) features
    hdf_io = h5py.File(PRISM_dir + 'PRISM_{}_features_2015_2020.hdf'.format(var), 'r')
    PRISM_T = hdf_io['{}_4km'.format(var)][ind_pred, ...]
    REGRID_T = hdf_io['{}_REGRID'.format(var)][ind_pred, ...]
    hdf_io.close()

    # import pre-trained models (import together for saving time)
    # UNET
    unet = {}
    unet['djf'] = keras.models.load_model(model_import_dir+'UNET3_{}_djf.hdf'.format(var))
    unet['mam'] = keras.models.load_model(model_import_dir+'UNET3_{}_mam.hdf'.format(var))
    unet['jja'] = keras.models.load_model(model_import_dir+'UNET3_{}_jja.hdf'.format(var))
    unet['son'] = keras.models.load_model(model_import_dir+'UNET3_{}_son.hdf'.format(var))
    for n, date in enumerate(pred_list):
        X = (REGRID_T[n, ...], etopo_4km, etopo_regrid)
        print(date)
        if date.month in [12, 1, 2]:
            temp_unet = vu.pred_domain(X, land_mask, unet['djf'], param, method='norm_std')
        elif date.month in [3, 4, 5]:
            temp_unet = vu.pred_domain(X, land_mask, unet['mam'], param, method='norm_std')
        elif date.month in [6, 7, 8]:
            temp_unet = vu.pred_domain(X, land_mask, unet['jja'], param, method='norm_std')
        elif date.month in [9, 10, 11]:
            temp_unet = vu.pred_domain(X, land_mask, unet['son'], param, method='norm_std')
    
        RESULT_UNET[n, ...] = temp_unet
            
    tuple_save = (lon_4km, lat_4km, PRISM_T, REGRID_T, RESULT_UNET)
    label_save = ['lon_4km', 'lat_4km', '{}_4km'.format(var), '{}_REGRID'.format(var), 'RESULT_UNET']
    du.save_hdf5(tuple_save, label_save, out_dir=save_dir, filename='PRISM_PRED_{}_test.hdf'.format(var))