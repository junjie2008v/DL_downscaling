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
# dt list for pred, 2018-2019
N_pred = 365
pred_list = [datetime(2018, 1, 1, 0) + timedelta(days=x) for x in range(N_pred)]
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
VARS = ['TMAX', 'TMIN']
seasons = ['djf', 'mam', 'jja', 'son']

for var in VARS:
    print('========== {} =========='.format(var))
    
    # allocation
    RESULT_UNET = np.zeros((N_pred,)+grid_shape)
    RESULT_UAE = np.zeros((N_pred,)+grid_shape)
    
    # import 3d (time, lat, lon) features
    with h5py.File(PRISM_dir + 'PRISM_{}_features_2015_2020.hdf'.format(var), 'r') as hdf_io:
        PRISM_T = hdf_io['{}_4km'.format(var)][ind_pred, ...]
        REGRID_T = hdf_io['{}_REGRID'.format(var)][ind_pred, ...]

    # import pre-trained models (import together for saving time)
    # UNET
    unet = {}
    unet['djf'] = keras.models.load_model(model_import_dir+'UNET3_{}_djf_tune.hdf'.format(var))
    unet['mam'] = keras.models.load_model(model_import_dir+'UNET3_{}_mam_tune.hdf'.format(var))
    unet['jja'] = keras.models.load_model(model_import_dir+'UNET3_{}_jja_tune.hdf'.format(var))
    unet['son'] = keras.models.load_model(model_import_dir+'UNET3_{}_son_tune.hdf'.format(var))
    # UNET-AE training/tuning domain
    uae_train = {}
    uae_train['djf'] = keras.models.load_model(model_import_dir+'UAE3_{}_djf_tune.hdf'.format(var))
    uae_train['mam'] = keras.models.load_model(model_import_dir+'UAE3_{}_mam_tune.hdf'.format(var))
    uae_train['jja'] = keras.models.load_model(model_import_dir+'UAE3_{}_jja_tune.hdf'.format(var))
    uae_train['son'] = keras.models.load_model(model_import_dir+'UAE3_{}_son_tune.hdf'.format(var))
    # UNET-AE transferring domain
    uae_trans = {}
    uae_trans['djf'] = keras.models.load_model(model_import_dir+'UAE3_{}_djf_trans2.hdf'.format(var))
    uae_trans['mam'] = keras.models.load_model(model_import_dir+'UAE3_{}_mam_trans2.hdf'.format(var))
    uae_trans['jja'] = keras.models.load_model(model_import_dir+'UAE3_{}_jja_trans2.hdf'.format(var))
    uae_trans['son'] = keras.models.load_model(model_import_dir+'UAE3_{}_son_trans2.hdf'.format(var))
    
    for n, date in enumerate(pred_list):
        X = (REGRID_T[n, ...], etopo_4km, etopo_regrid)
        print(date)
        if date.month in [12, 1, 2]:
            temp_unet = vu.pred_domain(X, land_mask, unet['djf'], param, method='norm_std')
            temp_uae_train = vu.pred_domain(X, land_mask, uae_train['djf'], param, method='norm_std')
            temp_uae_trans = vu.pred_domain(X, land_mask, uae_trans['djf'], param, method='norm_std')
        elif date.month in [3, 4, 5]:
            temp_unet = vu.pred_domain(X, land_mask, unet['mam'], param, method='norm_std')
            temp_uae_train = vu.pred_domain(X, land_mask, uae_train['mam'], param, method='norm_std')
            temp_uae_trans = vu.pred_domain(X, land_mask, uae_trans['mam'], param, method='norm_std')
        elif date.month in [6, 7, 8]:
            temp_unet = vu.pred_domain(X, land_mask, unet['jja'], param, method='norm_std')
            temp_uae_train = vu.pred_domain(X, land_mask, uae_train['jja'], param, method='norm_std')
            temp_uae_trans = vu.pred_domain(X, land_mask, uae_trans['jja'], param, method='norm_std')
        elif date.month in [9, 10, 11]:
            temp_unet = vu.pred_domain(X, land_mask, unet['son'], param, method='norm_std')
            temp_uae_train = vu.pred_domain(X, land_mask, uae_train['son'], param, method='norm_std')
            temp_uae_trans = vu.pred_domain(X, land_mask, uae_trans['son'], param, method='norm_std')
    
        RESULT_UNET[n, ...] = temp_unet
        RESULT_UAE[n, :ind_trans, :] = temp_uae_train[:ind_trans, :]
        RESULT_UAE[n, ind_trans:, :] = temp_uae_trans[ind_trans:, :]
        
    tuple_save = (lon_4km, lat_4km, PRISM_T, REGRID_T, RESULT_UNET, RESULT_UAE)
    label_save = ['lon_4km', 'lat_4km', '{}_4km'.format(var), '{}_REGRID'.format(var), 'RESULT_UNET', 'RESULT_UAE']
    du.save_hdf5(tuple_save, label_save, out_dir=save_dir, filename='PRISM_PRED_{}_2018_2020.hdf'.format(var))
    