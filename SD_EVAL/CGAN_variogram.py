# general tools
import sys
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import netCDF4 as nc
import numpy as np

# stats tools
from skgstat import Variogram
from scipy.interpolate import griddata

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/')
from namelist import *
import data_utils as du
import verif_utils as vu

DATA_dir = PRISM_dir
RESULT_dir = save_dir
era_feature_path = ERA_dir + 'ERA_TMEAN_features_2015_2020.hdf'
jra_feature_path = JRA_dir + 'JRA_TMEAN_features_2015_2020.hdf'
prism_path = PRISM_dir+'PRISM_regrid_2015_2020.hdf'
# result files
era_pred_name = 'ERA_TMEAN_clean'
jra_pred_name = 'JRA_TMEAN_clean'

TMEAN_key = 'TMEAN_025'
ERA_raw_key = 'era_raw'
JRA_raw_key = 'jra_raw'

with h5py.File(PRISM_dir+'land_mask_NA.hdf', 'r') as hdf_io:
    lon_clean = hdf_io['lon_025'][...]
    lat_clean = hdf_io['lat_025'][...]
    land_mask_clean = hdf_io['land_mask_025'][...]
    
# available time range of the file (2018-2019) <----- support 2020 test in the furture 
N_all = 365 + 366 + 365 + 365 + 365
all_list = [datetime(2015, 1, 1, 0) + timedelta(days=x) for x in range(N_all)]
# time range for plotting (2018-2019)
N_pred_era = 365 + 243
pred_list_era = [datetime(2018, 1, 1, 0) + timedelta(days=x) for x in range(N_pred_era)]
N_pred_jra = 365 + 365
pred_list_jra = [datetime(2018, 1, 1, 0) + timedelta(days=x) for x in range(N_pred_jra)]
# indices
ind_era = du.dt_match(all_list, pred_list_era)
ind_jra = du.dt_match(all_list, pred_list_jra)

with h5py.File(ERA_dir+era_pred_name+'_2018.hdf', 'r') as hdf_io:
    ERA_2018 = hdf_io[TMEAN_key][...]
with h5py.File(ERA_dir+era_pred_name+'_2019.hdf', 'r') as hdf_io:
    ERA_2019 = hdf_io[TMEAN_key][...]
    
with h5py.File(JRA_dir+jra_pred_name+'_2018.hdf', 'r') as hdf_io:
    JRA_2018 = hdf_io[TMEAN_key][...]
with h5py.File(JRA_dir+jra_pred_name+'_2019.hdf', 'r') as hdf_io:
    JRA_2019 = hdf_io[TMEAN_key][...]
    
ERA_clean = np.concatenate((ERA_2018, ERA_2019), axis=0)
JRA_clean = np.concatenate((JRA_2018, JRA_2019), axis=0)

ERA_clean[:, land_mask_clean] = np.nan
JRA_clean[:, land_mask_clean] = np.nan

with h5py.File(era_feature_path, 'r') as hdf_io:
    ERA_raw = hdf_io[ERA_raw_key][ind_era, ...]
    lon_ERA = hdf_io['lon_raw'][...]
    lat_ERA = hdf_io['lat_raw'][...]

with h5py.File(jra_feature_path, 'r') as hdf_io:
    JRA_raw = hdf_io[JRA_raw_key][ind_jra, ...]
    lon_JRA = hdf_io['lon_raw'][...]
    lat_JRA = hdf_io['lat_raw'][...]
    
# land mask interp
flag_mask = griddata((lon_clean.ravel(), lat_clean.ravel()), land_mask_clean.ravel(), (lon_ERA, lat_ERA), method='linear')
land_mask_ERA = flag_mask > 0
land_mask_ERA[:, :14] = True
land_mask_ERA[:, 71:] = True

flag_mask = griddata((lon_clean.ravel(), lat_clean.ravel()), land_mask_clean.ravel(), (lon_JRA, lat_JRA), method='linear')
land_mask_JRA = flag_mask > 0
land_mask_JRA[:, :18] = True
land_mask_JRA[:, 89:] = True

ERA_raw[:, land_mask_ERA] = np.nan
JRA_raw[:, land_mask_JRA] = np.nan

# ERA interp
grid_shape = lon_ERA.shape
ERA_clean_raw = np.empty((N_pred_era,)+grid_shape)
for i in range(N_pred_era):
    ERA_clean_raw[i, ...] = du.interp2d_wraper(lon_clean, lat_clean, ERA_clean[i, ...], lon_ERA, lat_ERA, method=interp_method)
ERA_clean_raw[:, land_mask_ERA] = np.nan

# JRA interp
grid_shape = lon_JRA.shape
JRA_clean_raw = np.empty((N_pred_jra,)+grid_shape)
for i in range(N_pred_jra):
    JRA_clean_raw[i, ...] = du.interp2d_wraper(lon_clean, lat_clean, JRA_clean[i, ...], lon_JRA, lat_JRA, method=interp_method)
JRA_clean_raw[:, land_mask_JRA] = np.nan


N_bins = 20
N_days_pred = {}
N_days_pred['ERA'] = N_pred_era # <------------------------
N_days_pred['JRA'] = N_pred_jra # <------------------------

land_mask = {}
land_mask['ERA'] = land_mask_ERA
land_mask['JRA'] = land_mask_JRA

latlon = {}
latlon['lon_ERA'] = lon_ERA
latlon['lat_ERA'] = lat_ERA
latlon['lon_JRA'] = lon_JRA
latlon['lat_JRA'] = lat_JRA

raw = {}
raw['ERA'] = ERA_raw
raw['JRA'] = JRA_raw
clean = {}
clean['ERA'] = ERA_clean_raw
clean['JRA'] = JRA_clean_raw

for s in ['ERA', 'JRA']:
    print('Working on {}'.format(s))
    N_days = N_days_pred[s] 
    distx, disty = du.latlon_to_dist(latlon['lon_{}'.format(s)], latlon['lat_{}'.format(s)])
    distx = distx*1e-3
    disty = disty*1e-3
    distx[land_mask[s]] = np.nan
    disty[land_mask[s]] = np.nan
    distx_clean = distx
    disty_clean = disty
    
#     distx_clean, disty_clean = du.latlon_to_dist(lon_clean, lat_clean)
#     distx_clean = distx_clean*1e-3
#     disty_clean = disty_clean*1e-3
#     distx_clean[land_mask_clean] = np.nan
#     disty_clean[land_mask_clean] = np.nan

    points = np.concatenate((distx[~land_mask[s]][:, None], 
                             disty[~land_mask[s]][:, None]), axis=1)
    points_clean = points
    #np.concatenate((distx_clean[~land_mask_clean][:, None], disty_clean[~land_mask_clean][:, None]), axis=1)
    
    N_points = len(points)
    N_points_clean = len(points_clean)


    points_repeat = np.repeat(points, N_days, axis=0)
    points_repeat_clean = np.repeat(points_clean, N_days, axis=0)
    
    val_raw_repeat = np.zeros(N_points*N_days)
    val_clean_repeat = np.zeros(N_points_clean*N_days)
    
    count = 0
    for i in range(N_days):
        val_raw = raw[s][i, ~land_mask[s]]
        val_clean = clean[s][i, ~land_mask[s]]#~land_mask_clean
        
        if i%30 == 0:
            val_raw_repeat[N_points*count:N_points*(count+1)] = val_raw
            val_clean_repeat[N_points_clean*count:N_points_clean*(count+1)] = val_clean
            count += 1
            
        V_raw = Variogram(points, val_raw, n_lags=N_bins, normalize=False)
        V_clean = Variogram(points_clean, val_clean, n_lags=N_bins, normalize=False)

        if i == 0:
            raw_dist = V_raw._dist
            clean_dist = V_clean._dist
            
            raw_diff = np.zeros([N_days, len(raw_dist)])
            clean_diff = np.zeros([N_days, len(clean_dist)])

        raw_diff[i, :] = V_raw._diff
        clean_diff[i, :] = V_clean._diff

    points_repeat = points_repeat[:N_points*count, ...]
    points_repeat_clean = points_repeat[:N_points_clean*count, ...]
    
    
    val_raw_repeat = val_raw_repeat[:N_points*count, ...]
    val_clean_repeat = val_clean_repeat[:N_points_clean*count, ...]

    V_raw = Variogram(points_repeat, val_raw_repeat, n_lags=N_bins, normalize=False)
    V_clean = Variogram(points_repeat_clean, val_clean_repeat, n_lags=N_bins, normalize=False)

    bins_raw = V_raw._bins
    exp_raw = V_raw.experimental
    bins_clean = V_clean._bins
    exp_clean = V_clean.experimental

    x_raw = np.linspace(0, bins_raw[-1], 1000)
    y_raw = V_raw.transform(x_raw)
    x_clean = np.linspace(0, bins_clean[-1], 1000)
    y_clean = V_clean.transform(x_clean)

    save_tuple = (land_mask[s], raw_dist, raw_diff, bins_raw, exp_raw, x_raw, y_raw, 
                  clean_dist, clean_diff, bins_clean, exp_clean, x_clean, y_clean)
    save_label = ['land_mask', 'raw_dist', 'raw_diff', 'bins_raw', 'exp_raw', 'fitx_raw', 'fity_raw', 
                  'clean_dist', 'clean_diff', 'bins_clean', 'exp_clean', 'fitx_clean', 'fity_clean']
    du.save_hdf5(save_tuple, save_label, save_dir, 'CGAN_{}_Variogram_backup.hdf'.format(s))