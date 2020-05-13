import sys
import h5py
import numpy as np
from glob import glob
from tensorflow import keras
from datetime import datetime, timedelta
#
sys.path.insert(0, '/glade/u/home/ksha/ML_repo/utils/')
import PRISM_utils as pu
import data_utils as du
import keras_utils as ku
from namelist_PRISM import * 
#
VAR_list = ['TMAX', 'TMIN']

L = 366+365+364 # 2016-2019

cut = 504
edge = 32; gap = 8; size = 128; size_center = size-2*edge

save_dir = '/glade/work/ksha/data/Keras/PRISM_publish/'
mlr_dir = save_dir +'MLR_T2_NCEP.hdf'

label = []; data = []

base = datetime(2016, 1, 1, 0)
date_list = [base + timedelta(days=x) for x in range(L)]

for VAR in VAR_list:
    print(VAR)
    # model
    # a
    model_unet_a1 = keras.models.load_model(save_dir+'UNET_{}_A_djf_tune.hdf'.format(VAR))
    model_unet_a2 = keras.models.load_model(save_dir+'UNET_{}_A_mam_tune.hdf'.format(VAR))
    model_unet_a3 = keras.models.load_model(save_dir+'UNET_{}_A_jja_tune.hdf'.format(VAR))
    model_unet_a4 = keras.models.load_model(save_dir+'UNET_{}_A_son_tune.hdf'.format(VAR))
    # b
    model_unet_b1 = keras.models.load_model(save_dir+'UNET_{}_B_djf_tune_train.hdf'.format(VAR))
    model_unet_b2 = keras.models.load_model(save_dir+'UNET_{}_B_mam_tune_train.hdf'.format(VAR))
    model_unet_b3 = keras.models.load_model(save_dir+'UNET_{}_B_jja_tune_train.hdf'.format(VAR))
    model_unet_b4 = keras.models.load_model(save_dir+'UNET_{}_B_son_tune_train.hdf'.format(VAR))
    # b
    model_unet_c1 = keras.models.load_model(save_dir+'UNET_{}_B_djf_tune_trans.hdf'.format(VAR))
    model_unet_c2 = keras.models.load_model(save_dir+'UNET_{}_B_mam_tune_trans.hdf'.format(VAR))
    model_unet_c3 = keras.models.load_model(save_dir+'UNET_{}_B_jja_tune_trans.hdf'.format(VAR))
    model_unet_c4 = keras.models.load_model(save_dir+'UNET_{}_B_son_tune_trans.hdf'.format(VAR))
    # std data
    hdf_io = h5py.File(BACKUP_dir+'NCEP_std_2016_2018_4km.hdf', 'r')
    MEAN = hdf_io['FIT_{}'.format(VAR)][...]
    STD  = hdf_io['STD_{}'.format(VAR)][...]
    REGRID_VAR = hdf_io['{}_REGRID'.format(VAR)][...]
    hdf_io.close()
    # etopo
    hdf_io = h5py.File(BACKUP_dir+'PRISM_cubic_2015_2018.hdf', 'r')
    land_mask = np.isnan(hdf_io['PRISM_{}'.format(VAR)][0, ...])
    T_TRUE = hdf_io['PRISM_{}'.format(VAR)][365:-2, ...]
    etopo = hdf_io['etopo'][...]
    etopo_regrid = hdf_io['etopo_regrid'][...]
    hdf_io.close()
    print('{} v.s. {}'.format(len(T_TRUE), L))
    # ETOPO preprocess
    IN_ETOPO, IN_DIFF = pu.feature_pre(etopo, etopo_regrid, land_mask)
    # MLR 
    hdf_io = h5py.File(mlr_dir)
    T_MLR = hdf_io['{}_MLR'.format(VAR)][...]
    hdf_io.close()
    #
    MEAN[:, land_mask] = np.nan
    RESULT_a = np.empty(REGRID_VAR.shape)
    RESULT_b = np.empty(REGRID_VAR.shape)
    #
    for n, date in enumerate(date_list):
        print('\t {}'.format(n))
        IN = REGRID_VAR[n, ...]
        if date.month in [12, 1, 2]:
            dscale_unet_a = pu.feature_to_domain(IN, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_a1, ind=0)
            dscale_unet_b = pu.feature_to_domain(IN, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_b1, ind=0)
            dscale_unet_c = pu.feature_to_domain(IN, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_c1, ind=0)
        elif date.month in [3, 4, 5]:
            dscale_unet_a = pu.feature_to_domain(IN, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_a2, ind=0)
            dscale_unet_b = pu.feature_to_domain(IN, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_b2, ind=0)
            dscale_unet_c = pu.feature_to_domain(IN, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_c2, ind=0)
        elif date.month in [6, 7, 8]:
            dscale_unet_a = pu.feature_to_domain(IN, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_a3, ind=0)
            dscale_unet_b = pu.feature_to_domain(IN, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_b3, ind=0)
            dscale_unet_c = pu.feature_to_domain(IN, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_c3, ind=0)
        elif date.month in [9, 10, 11]:
            dscale_unet_a = pu.feature_to_domain(IN, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_a4, ind=0)
            dscale_unet_b = pu.feature_to_domain(IN, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_b4, ind=0)
            dscale_unet_c = pu.feature_to_domain(IN, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_c4, ind=0)
        RESULT_a[n, ...] = dscale_unet_a
        #
        RESULT_b[n, :cut, :] = dscale_unet_b[:cut, :]
        RESULT_b[n, cut:, :] = dscale_unet_c[cut:, :]
    # norm inv    
    RESULT_a = RESULT_a*STD+MEAN
    RESULT_b = RESULT_b*STD+MEAN
    T_MLR = T_MLR*STD+MEAN
    T_REGRID = REGRID_VAR*STD+MEAN
    # append
    data.append(T_TRUE)
    data.append(T_REGRID)
    data.append(RESULT_a)
    data.append(RESULT_b)
    data.append(T_MLR)
    label += ['{}_TRUE'.format(VAR), '{}_REGRID'.format(VAR), '{}_A'.format(VAR), '{}_B'.format(VAR), '{}_MLR'.format(VAR)]
    
du.save_hdf5(tuple(data), label, out_dir=save_dir, filename='PRISM_PRED_NCEP_2016_2018_sea.hdf')