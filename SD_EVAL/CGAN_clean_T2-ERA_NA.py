# general tools
import sys
import time
import argparse
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

# parse user inputs
parser = argparse.ArgumentParser()
# positionals
parser.add_argument('y', help='year')
args = vars(parser.parse_args())
year = int(args['y'])

# defining prediction range
# dt list for all
N_all = 365 + 366 + 365 + 365 + 243
all_list = [datetime(2015, 1, 1, 0) + timedelta(days=x) for x in range(N_all)]
# pred range
if year == 2019:
    N_pred = 243
else:
    N_pred = 365
pred_list = [datetime(year, 1, 1, 0) + timedelta(days=x) for x in range(N_pred)]
# indices
ind_pred = du.dt_match(all_list, pred_list)


# import geo. info
with h5py.File(PRISM_dir+'land_mask_NA.hdf', 'r') as hdf_io:
    lon_4km = hdf_io['lon_4km'][...]
    lat_4km = hdf_io['lat_4km'][...]
    lon_025 = hdf_io['lon_025'][...]
    lat_025 = hdf_io['lat_025'][...]
    etopo_4km = hdf_io['etopo_4km'][...]
    etopo_regrid = hdf_io['etopo_regrid'][...]
    land_mask = hdf_io['land_mask'][...]

grid_shape = land_mask.shape

# macros
# cycle-GAN import
model_import_dir = temp_dir
CGAN = keras.models.load_model(model_import_dir+'Cycle-GAN_TMEAN_LR.hdf')

# overlapped tile prediction settings
param = {}
param['gap'] = 22
param['edge'] = 44
param['size'] = 96

# loop over variables and seasons
VARS = ['TMEAN']

for VAR in VARS:
    # import 3d (time, lat, lon) features
    with h5py.File(ERA_dir + 'ERA_{}_features_2015_2020.hdf'.format(VAR), 'r') as hdf_io:
        REGRID_T = hdf_io['{}_REGRID'.format(VAR)][ind_pred, ...]
        
    shape_3d = REGRID_T.shape
    RESULT_CLEAN = np.zeros(shape_3d)
    RESULT_025 = np.zeros((shape_3d[0],)+lon_025.shape)
    
    for n in range(shape_3d[0]):
        #start_time = time.time()
        print('\t{}'.format(n))
        X = (REGRID_T[n, ...], etopo_regrid)
        temp_unet = vu.pred_domain(X, land_mask, CGAN, param, method='norm_std')
        #temp_025 = du.nearest_wraper(lon_4km, lat_4km, temp_unet, lon_025, lat_025)
        temp_025 = du.interp2d_wraper(lon_4km, lat_4km, temp_unet, lon_025, lat_025, method=interp_method)
        temp_4km = du.interp2d_wraper(lon_025, lat_025, temp_025, lon_4km, lat_4km, method=interp_method)
        RESULT_025[n, ...] = temp_025
        RESULT_CLEAN[n, ...] = temp_4km
        #print("--- %s seconds ---" % (time.time() - start_time))
    RESULT_CLEAN[:, land_mask] = np.nan
    tuple_save = (lon_4km, lat_4km, RESULT_CLEAN, RESULT_025, etopo_4km, etopo_regrid)
    label_save = ['lon_4km', 'lat_4km', '{}_REGRID'.format(VAR), '{}_025'.format(VAR), 'etopo_4km', 'etopo_regrid']
    du.save_hdf5(tuple_save, label_save, out_dir=ERA_dir, filename='ERA_{}_clean_{}.hdf'.format(VAR, year))
    