import sys
import h5py
import numpy as np
from glob import glob
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/ML_repo/utils/')
import PRISM_utils as pu
import data_utils as du
import keras_utils as ku
import graph_utils as gu
from namelist_PRISM import * 
# 
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from joblib import dump, load
# sklearn model 2d application
def ir_apply(sklearn_model, grid):
    out = np.zeros(grid.shape)*np.nan
    pick_flag = ~np.isnan(grid)
    out[pick_flag] = sklearn_model.transform(grid[pick_flag])
    return out

save_dir = '/glade/work/ksha/data/Keras/PRISM_publish/'
print('Reliability diagram fit')
train_L = 365+366+365; N=500; cut = 504 # training domain index
# import base data
hdf_io = h5py.File(BACKUP_dir+'PRISM_pct_2015_2018.hdf', 'r')
PRISM_TRAIN = hdf_io['PRISM_PCT'][:train_L, ...]
hdf_io.close()
# land mask
land_mask = np.isnan(PRISM_TRAIN[0])
temp_flag = PRISM_TRAIN > 0 # True=nonzero_precip # <---- nan warnings 
# fit by seasons
seasons = ['DJF', 'MAM', 'JJA', 'SON']
base = datetime(2015, 1, 1, 0)
date_list = [base + timedelta(days=x) for x in range(train_L)]
IND = {}; L = len(date_list)
base_ind0 = np.zeros(L).astype(bool)
base_ind1 = np.zeros(L).astype(bool)
base_ind2 = np.zeros(L).astype(bool)
base_ind3 = np.zeros(L).astype(bool)
for i, dt in enumerate(date_list):
    m = dt.month
    if m==1 or m==2 or m==12:
        base_ind0[i] = True
    elif m==3 or m==4 or m==5:
        base_ind1[i] = True
    elif m==6 or m==7 or m==8:
        base_ind2[i] = True
    else:
        base_ind3[i] = True
IND[seasons[0]] = base_ind0
IND[seasons[1]] = base_ind1
IND[seasons[2]] = base_ind2
IND[seasons[3]] = base_ind3
# loop over cases
models = ['PCT_REGRID', 'UNET_A', 'XNET_A']
reli_diagram = {}
for i, model in enumerate(models):
    print('Model: {}'.format(model))
    # get training set pred
    hdf_io = h5py.File(save_dir+'PRISM_PRED_PCT_2015_2018.hdf', 'r')
    UNET_TRAIN = hdf_io[model][:train_L, ...]
    hdf_io.close()
    UNET_TRAIN[:, land_mask] = np.nan
    # min-max norm
    temp_min  = np.nanmin(UNET_TRAIN, axis=(1, 2))
    temp_max  = np.nanmax(UNET_TRAIN, axis=(1, 2))
    temp_norm = (UNET_TRAIN[:, :cut, :]-temp_min[:, None, None])/(temp_max-temp_min)[:, None, None]
    for sea in seasons:
        print('\tSeason: {}'.format(sea))
        # seasons indexing
        flat_true = temp_flag[IND[sea], :cut, :].flatten() # flatten contains copy
        flat_norm = temp_norm[IND[sea], :cut, :].flatten()
        pick_flag = ~np.isnan(flat_norm)
        flat_true = flat_true[pick_flag]
        flat_norm = flat_norm[pick_flag]
        # reliability diagram
        prob_true, prob_pred = calibration_curve(flat_true, flat_norm, n_bins=N, strategy='quantile')
        reli_diagram['{}_{}'.format(model, sea)] = np.concatenate((prob_true[:, None], prob_pred[:, None]), axis=1)
        # isotonic regress.
        IR = IsotonicRegression(out_of_bounds='clip')
        IR.fit(prob_pred, prob_true)
        # Save model
        dump(IR, save_dir+'IR_{}_{}.sklearn'.format(model, sea))
# save history
np.save(save_dir+'IR_curve.npy', reli_diagram)
# ========== #
print('Estimating precip probabilities')
test_L = 366
# hdf_io = h5py.File(BACKUP_dir+'PRISM_pct_2015_2018.hdf', 'r')
# land_mask = np.isnan(hdf_io['PRISM_PCT'][0, ...])
# hdf_io.close()
grid_shape = land_mask.shape
base = datetime(2018, 1, 1, 0)
date_list = [base + timedelta(days=x) for x in range(test_L)]

RESULT = {}
models = ['PCT_REGRID', 'UNET_A', 'UNET_B', 'XNET_A', 'XNET_B']
for model in models:
    print('Model: {}'.format(model))
    # get training set pred
    hdf_io = h5py.File(save_dir+'PRISM_PRED_PCT_2015_2018.hdf', 'r')
    UNET_TEST = hdf_io[model][-test_L:, ...]
    hdf_io.close()
    UNET_TEST[:, land_mask] = np.nan
    # min-max norm
    temp_min  = np.nanmin(UNET_TEST, axis=(1, 2))
    temp_max  = np.nanmax(UNET_TEST, axis=(1, 2))
    temp_norm = (UNET_TEST-temp_min[:, None, None])/(temp_max-temp_min)[:, None, None]
    #
    out = np.zeros(temp_norm.shape)*np.nan
    IR1 = load(save_dir+'IR_{}_DJF.sklearn'.format(model))
    IR2 = load(save_dir+'IR_{}_MAM.sklearn'.format(model))
    IR3 = load(save_dir+'IR_{}_JJA.sklearn'.format(model))
    IR4 = load(save_dir+'IR_{}_SON.sklearn'.format(model))
    for i, date in enumerate(date_list):
        if i%60 == 0:
            print(date) # monitoring the progress
        if date.month in [12, 1, 2]:
            out[i, ...] = ir_apply(IR1, temp_norm[i, ...])
        elif date.month in [3, 4, 5]:
            out[i, ...] = ir_apply(IR2, temp_norm[i, ...])
        elif date.month in [6, 7, 8]:
            out[i, ...] = ir_apply(IR3, temp_norm[i, ...])
        elif date.month in [9, 10, 11]:
            out[i, ...] = ir_apply(IR4, temp_norm[i, ...])
    RESULT[model] = out
# save output
du.save_hdf5(tuple(RESULT.values()), models, save_dir, 'PRISM_PCT_flags_2018.hdf')


