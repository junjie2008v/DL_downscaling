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
base = datetime(2016, 1, 1, 0)
date_list = [base + timedelta(days=x) for x in range(366+365+364)] # 2015-01-01 to 2018-12-30
# labels
label = []; data = []
# UNet
print('UNet model')
model_unet_a1 = keras.models.load_model(save_dir+'UNET_PCT_A_djf.hdf')
model_unet_a2 = keras.models.load_model(save_dir+'UNET_PCT_A_mam.hdf')
model_unet_a3 = keras.models.load_model(save_dir+'UNET_PCT_A_jja.hdf')
model_unet_a4 = keras.models.load_model(save_dir+'UNET_PCT_A_son.hdf')
# UNet-AE
print('UNet AE')
model_unet_b1 = keras.models.load_model(save_dir+'UNET_PCT_B_djf.hdf')
model_unet_b2 = keras.models.load_model(save_dir+'UNET_PCT_B_mam.hdf')
model_unet_b3 = keras.models.load_model(save_dir+'UNET_PCT_B_jja.hdf')
model_unet_b4 = keras.models.load_model(save_dir+'UNET_PCT_B_son.hdf')
# XNet
print('Nested UNet')
model_unet_x1 = keras.models.load_model(save_dir+'XNET_PCT_djf.hdf')
model_unet_x2 = keras.models.load_model(save_dir+'XNET_PCT_mam.hdf')
model_unet_x3 = keras.models.load_model(save_dir+'XNET_PCT_jja.hdf')
model_unet_x4 = keras.models.load_model(save_dir+'XNET_PCT_son.hdf')
# XNet-AE
print('Nested UNet AE')
model_unet_y1 = keras.models.load_model(save_dir+'XNET_PCT_B_djf.hdf')
model_unet_y2 = keras.models.load_model(save_dir+'XNET_PCT_B_mam.hdf')
model_unet_y3 = keras.models.load_model(save_dir+'XNET_PCT_B_jja.hdf')
model_unet_y4 = keras.models.load_model(save_dir+'XNET_PCT_B_son.hdf')
# std data
h_io = h5py.File('/glade/scratch/ksha/BACKUP/NCEP_FNL_BC_2016_2018_4km.hdf', 'r')
lon = h_io['lon'][...]
lat = h_io['lat'][...]
REGRID_PCT = h_io['PCT_REGRID'][...]
etopo = h_io['etopo_BC'][...]
CLIM_PCT = h_io['CLIM_PRISM_PCT'][...]
land_mask = h_io['land_mask'][...]
h_io.close()

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
data.append(REGRID_PCT)
data.append(RESULT_a)
data.append(RESULT_b)
data.append(RESULT_x)
data.append(RESULT_y)
label += ['PCT_REGRID', 'UNET_A', 'UNET_B', 'XNET_A', 'XNET_B']

du.save_hdf5(tuple(data), label, out_dir=save_dir, filename='PRISM_PRED_NCEP_PCT_BC_2016_2018.hdf')