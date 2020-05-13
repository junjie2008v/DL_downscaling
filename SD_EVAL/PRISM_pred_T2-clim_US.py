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
# time range
valid_range = [[datetime(2017, 2, 5)  , datetime(2017, 3, 5)],
               [datetime(2017, 4, 11) , datetime(2017, 5, 11)],
               [datetime(2017, 6, 4)  , datetime(2017, 7, 4)],
               [datetime(2017, 10, 25), datetime(2017, 11, 25)]]
test_range  = [[datetime(2018, 1, 1), datetime(2018, 3, 1)],
               [datetime(2018, 3, 2), datetime(2018, 6, 1)],
               [datetime(2018, 6, 2), datetime(2018, 9, 1)],
               [datetime(2018, 9, 2), datetime(2018, 12, 31)]]

hdf_io_temp = h5py.File(BACKUP_dir+'PRISM_TMAX_2015_2018.hdf', 'r')
datenum = hdf_io_temp['datenum'][...]
hdf_io_temp.close()

_, _, test_ind = pu.subset_valid(datenum, valid_range, test_range)
L = len(test_ind)

cut = 504
edge = 32; gap = 8; size = 128; size_center = size-2*edge

out_dir = '/glade/work/ksha/data/Keras/PRISM_publish/'
save_dir = '/glade/work/ksha/data/Keras/BACKUP/'
mlr_dir = save_dir +'MLR_T2_std.hdf'

base = datetime(2018, 1, 1, 0)
date_list = [base + timedelta(days=x) for x in range(365)]

label = []; data = []

for VAR in VAR_list:
    print(VAR)
    # model
    # a
    model_unet_a1 = keras.models.load_model(save_dir+'UNET_{}_A4_djf.hdf'.format(VAR))
    model_unet_a2 = keras.models.load_model(save_dir+'UNET_{}_A4_mam.hdf'.format(VAR))
    model_unet_a3 = keras.models.load_model(save_dir+'UNET_{}_A4_jja.hdf'.format(VAR))
    model_unet_a4 = keras.models.load_model(save_dir+'UNET_{}_A4_son.hdf'.format(VAR))
    # b
    model_unet_b1 = keras.models.load_model(save_dir+'UNET_{}_B4_djf_tune_train.hdf'.format(VAR))
    model_unet_b2 = keras.models.load_model(save_dir+'UNET_{}_B4_mam_tune_train.hdf'.format(VAR))
    model_unet_b3 = keras.models.load_model(save_dir+'UNET_{}_B4_jja_tune_train.hdf'.format(VAR))
    model_unet_b4 = keras.models.load_model(save_dir+'UNET_{}_B4_son_tune_train.hdf'.format(VAR))
    # b
    model_unet_c1 = keras.models.load_model(save_dir+'UNET_{}_B4_djf_tune_trans.hdf'.format(VAR))
    model_unet_c2 = keras.models.load_model(save_dir+'UNET_{}_B4_mam_tune_trans.hdf'.format(VAR))
    model_unet_c3 = keras.models.load_model(save_dir+'UNET_{}_B4_jja_tune_trans.hdf'.format(VAR))
    model_unet_c4 = keras.models.load_model(save_dir+'UNET_{}_B4_son_tune_trans.hdf'.format(VAR))
    # std data    
    hdf_io = h5py.File(BACKUP_dir+'PRISM_std_clim_2015_2018.hdf', 'r')
    PRISM_VAR = hdf_io['PRISM_{}'.format(VAR)][test_ind, ...]
    REGRID_VAR = hdf_io['{}_REGRID'.format(VAR)][test_ind, ...]
    CLIM_VAR = hdf_io['{}_CLIM'.format(VAR)][test_ind, ...]
    etopo = hdf_io['etopo'][...]
    etopo_regrid = hdf_io['etopo_regrid'][...]
    lon = hdf_io['lon'][...]
    lat = hdf_io['lat'][...]
    hdf_io.close()
    # land mask
    land_mask = np.isnan(PRISM_VAR[0, ...])
    # ETOPO preprocess
    IN_ETOPO, IN_DIFF = pu.feature_pre(etopo, etopo_regrid, land_mask)
    # MLR by seasons
    hdf_io = h5py.File('/glade/work/ksha/data/Keras/PRISM_publish/MLR_T2_std_season.hdf')
    T_MLR = hdf_io['{}_MLR'.format(VAR)][test_ind, ...]
    hdf_io.close()
    hdf_io = h5py.File(BACKUP_dir+'PRISM_std_2015_2018.hdf', 'r')
    T_MEAN = hdf_io['FIT_{}'.format(VAR)][test_ind, ...]
    T_STD  = hdf_io['STD_{}'.format(VAR)][test_ind, ...]
    hdf_io.close()
    T_MEAN[:, land_mask] = np.nan
    T_MLR = T_MLR*T_STD+T_MEAN
    #
    RESULT_a = np.empty(REGRID_VAR.shape)
    RESULT_b = np.empty(REGRID_VAR.shape)
    #L
    for n, date in enumerate(date_list):
        print(date)
        IN = REGRID_VAR[n, ...]
        CLIM = CLIM_VAR[n, ...]
        
        if date.month in [12, 1, 2]:
            dscale_unet_a = pu.feature_to_domain(IN, CLIM, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_a1, ind=0)
            dscale_unet_b = pu.feature_to_domain(IN, CLIM, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_b1, ind=0)
            dscale_unet_c = pu.feature_to_domain(IN, CLIM, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_c1, ind=0)
        elif date.month in [3, 4, 5]:
            dscale_unet_a = pu.feature_to_domain(IN, CLIM, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_a2, ind=0)
            dscale_unet_b = pu.feature_to_domain(IN, CLIM, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_b2, ind=0)
            dscale_unet_c = pu.feature_to_domain(IN, CLIM, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_c2, ind=0)
        elif date.month in [6, 7, 8]:
            dscale_unet_a = pu.feature_to_domain(IN, CLIM, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_a3, ind=0)
            dscale_unet_b = pu.feature_to_domain(IN, CLIM, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_b3, ind=0)
            dscale_unet_c = pu.feature_to_domain(IN, CLIM, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_c3, ind=0)
        elif date.month in [9, 10, 11]:
            dscale_unet_a = pu.feature_to_domain(IN, CLIM, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_a4, ind=0)
            dscale_unet_b = pu.feature_to_domain(IN, CLIM, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_b4, ind=0)
            dscale_unet_c = pu.feature_to_domain(IN, CLIM, IN_ETOPO, IN_DIFF, land_mask, size, edge, gap, model_unet_c4, ind=0)
        RESULT_a[n, ...] = dscale_unet_a
        #
        RESULT_b[n, :cut, :] = dscale_unet_b[:cut, :]
        RESULT_b[n, cut:, :] = dscale_unet_c[cut:, :]
    # append
    data.append(PRISM_VAR)
    data.append(REGRID_VAR)
    data.append(RESULT_a)
    data.append(RESULT_b)
    data.append(T_MLR)
    label += ['{}_TRUE'.format(VAR), '{}_REGRID'.format(VAR), '{}_A'.format(VAR), '{}_B'.format(VAR), '{}_MLR'.format(VAR)]
    
du.save_hdf5(tuple(data), label, out_dir=out_dir, filename='PRISM_PRED_T2_2018_clim.hdf')