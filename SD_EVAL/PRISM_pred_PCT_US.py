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
# param
cut = 504
edge = 32; gap = 8; size = 128; size_center = size-2*edge
save_dir = '/glade/work/ksha/data/Keras/PRISM_publish/'
# datetime
base = datetime(2015, 1, 1, 0)
date_list = [base + timedelta(days=x) for x in range(365+366+365+365+1)] # 2015-01-01 to 2019-01-01 365+366+365+365+1
# labels
label = []; data = []
# UNet
model_unet_a1 = keras.models.load_model(save_dir+'UNET_PCT_A_djf.hdf')
model_unet_a2 = keras.models.load_model(save_dir+'UNET_PCT_A_mam.hdf')
model_unet_a3 = keras.models.load_model(save_dir+'UNET_PCT_A_jja.hdf')
model_unet_a4 = keras.models.load_model(save_dir+'UNET_PCT_A_son.hdf')
# UNet-AE
model_unet_b1 = keras.models.load_model(save_dir+'UNET_PCT_B_djf.hdf')
model_unet_b2 = keras.models.load_model(save_dir+'UNET_PCT_B_mam.hdf')
model_unet_b3 = keras.models.load_model(save_dir+'UNET_PCT_B_jja.hdf')
model_unet_b4 = keras.models.load_model(save_dir+'UNET_PCT_B_son.hdf')
# XNet
model_unet_x1 = keras.models.load_model(save_dir+'XNET_PCT_djf.hdf')
model_unet_x2 = keras.models.load_model(save_dir+'XNET_PCT_mam.hdf')
model_unet_x3 = keras.models.load_model(save_dir+'XNET_PCT_jja.hdf')
model_unet_x4 = keras.models.load_model(save_dir+'XNET_PCT_son.hdf')
# XNet-AE
model_unet_y1 = keras.models.load_model(save_dir+'XNET_PCT_B_djf.hdf')
model_unet_y2 = keras.models.load_model(save_dir+'XNET_PCT_B_mam.hdf')
model_unet_y3 = keras.models.load_model(save_dir+'XNET_PCT_B_jja.hdf')
model_unet_y4 = keras.models.load_model(save_dir+'XNET_PCT_B_son.hdf')
# std data
hdf_io = h5py.File(BACKUP_dir+'PRISM_pct_2015_2018.hdf', 'r')
PRISM_PCT = hdf_io['PRISM_PCT'][...]
REGRID_PCT = hdf_io['PCT_REGRID'][...]
CLIM_PCT = hdf_io['CLIM_PRISM_PCT'][...]
etopo = hdf_io['etopo'][...]
lon = hdf_io['lon'][...]
lat = hdf_io['lat'][...]
hdf_io.close()
# landmask
land_mask = np.isnan(PRISM_PCT[0, ...])
# ETOPO preprocess
IN_ETOPO = np.copy(etopo)
IN_ETOPO[IN_ETOPO<0]=0 # ocean grid point match done within PRISM_utils


RESULT_a = np.empty(REGRID_PCT.shape)
RESULT_b = np.empty(REGRID_PCT.shape)
RESULT_x = np.empty(REGRID_PCT.shape)
RESULT_y = np.empty(REGRID_PCT.shape)
#L
for n, date in enumerate(date_list):
    print(date)
    IN = REGRID_PCT[n, ...]
    CLIM = CLIM_PCT[n, ...]
    if date.month in [12, 1, 2]:
        dscale_a = pu.feature_to_domain_pct(IN, CLIM, IN_ETOPO, land_mask, size, edge, gap, model_unet_a1, ind=0)
        dscale_b = pu.feature_to_domain_pct(IN, CLIM, IN_ETOPO, land_mask, size, edge, gap, model_unet_b1, ind=0)
        dscale_x = pu.feature_to_domain_pct(IN, CLIM, IN_ETOPO, land_mask, size, edge, gap, model_unet_x1, ind=0)
        dscale_y = pu.feature_to_domain_pct(IN, CLIM, IN_ETOPO, land_mask, size, edge, gap, model_unet_y1, ind=0)
    elif date.month in [3, 4, 5]:
        dscale_a = pu.feature_to_domain_pct(IN, CLIM, IN_ETOPO, land_mask, size, edge, gap, model_unet_a2, ind=0)
        dscale_b = pu.feature_to_domain_pct(IN, CLIM, IN_ETOPO, land_mask, size, edge, gap, model_unet_b2, ind=0)
        dscale_x = pu.feature_to_domain_pct(IN, CLIM, IN_ETOPO, land_mask, size, edge, gap, model_unet_x2, ind=0)
        dscale_y = pu.feature_to_domain_pct(IN, CLIM, IN_ETOPO, land_mask, size, edge, gap, model_unet_y2, ind=0)
    elif date.month in [6, 7, 8]:
        dscale_a = pu.feature_to_domain_pct(IN, CLIM, IN_ETOPO, land_mask, size, edge, gap, model_unet_a3, ind=0)
        dscale_b = pu.feature_to_domain_pct(IN, CLIM, IN_ETOPO, land_mask, size, edge, gap, model_unet_b3, ind=0)
        dscale_x = pu.feature_to_domain_pct(IN, CLIM, IN_ETOPO, land_mask, size, edge, gap, model_unet_x3, ind=0)
        dscale_y = pu.feature_to_domain_pct(IN, CLIM, IN_ETOPO, land_mask, size, edge, gap, model_unet_y3, ind=0)
    elif date.month in [9, 10, 11]:
        dscale_a = pu.feature_to_domain_pct(IN, CLIM, IN_ETOPO, land_mask, size, edge, gap, model_unet_a4, ind=0)
        dscale_b = pu.feature_to_domain_pct(IN, CLIM, IN_ETOPO, land_mask, size, edge, gap, model_unet_b4, ind=0)
        dscale_x = pu.feature_to_domain_pct(IN, CLIM, IN_ETOPO, land_mask, size, edge, gap, model_unet_x4, ind=0)
        dscale_y = pu.feature_to_domain_pct(IN, CLIM, IN_ETOPO, land_mask, size, edge, gap, model_unet_y4, ind=0)
    RESULT_a[n, ...] = dscale_a
    RESULT_b[n, ...] = dscale_b
    RESULT_x[n, ...] = dscale_x
    RESULT_y[n, ...] = dscale_y
# append
data.append(PRISM_PCT)
data.append(REGRID_PCT)
data.append(RESULT_a)
data.append(RESULT_b)
data.append(RESULT_x)
data.append(RESULT_y)
label += ['PRISM_PCT', 'PCT_REGRID', 'UNET_A', 'UNET_B', 'XNET_A', 'XNET_B']
    
du.save_hdf5(tuple(data), label, out_dir=save_dir, filename='PRISM_PRED_PCT_2015_2018.hdf')