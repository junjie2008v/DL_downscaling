import sys
from os.path import basename
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np

import metpy.calc
from metpy.units import units
from numpy import nansum, nanmax, nanmin, nanmean
import netCDF4 as nc

# geo tools
import shapely
from scipy.interpolate import griddata
from cartopy.io.shapereader import Reader

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/utils/')
import data_utils as du
import verif_utils as vu
from namelist import * 

def center_point(x1, y1, x2, y2, xi):
    return y1 + (y2-y1) * (xi-x1) / (x2-x1)

def linear_slope(x, y):
    X = x - x.mean()
    Y = y - y.mean()
    slope = (X.dot(Y)) / (X.dot(X))
    return slope

print('Import lat/lon and land mask')
with h5py.File(PRISM_dir+'land_mask_NA.hdf', 'r') as hdf_io:
    lon_4km = hdf_io['lon_4km'][...]
    lat_4km = hdf_io['lat_4km'][...]
    lon_025_clean = hdf_io['lon_025'][...]
    lat_025_clean = hdf_io['lat_025'][...]
    etopo_regrid_clean = hdf_io['etopo_regrid'][...]
    land_mask = hdf_io['land_mask'][...]

print('Raw netCDF4 process')
with nc.Dataset(ERA_TAIR_dir + 'ei.oper.an.ml.regn128sc.2019083118.sha416825.nc', 'r') as nc_io:
    A = nc_io['lv_HYBL1_a'][...]
    B = nc_io['lv_HYBL1_b'][...]
    x025 = nc_io['g4_lon_5'][...]-360
    y025 = nc_io['g4_lat_4'][...]
lon_025, lat_025 = np.meshgrid(x025, y025)

A = A[::-1][:20]
B = B[::-1][:20]

# nc variable keys
nc_keys = {}
nc_keys['TMEAN'] = 'T_GDS4_HYBL'
nc_keys['SFP'] = 'SP_GDS4_SFC'
# filenames
filenames = {}
filenames['TMEAN'] = sorted(glob(ERA_TAIR_dir+'*.nc'))
filenames['SFP'] = sorted(glob(ERA_SFP_dir+'*.nc'))

# aggregation method
method = 'nanmean'
# datetime info
days_ref = 365 + 243

# array size
levs = 20 # veritial levels
freq = 4 # 4 times a day
grid_shape = lon_025.shape

mon_days_365 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
mon_days_366 = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

print('Processing air temp')
VAR = 'TMEAN'
f = eval(method)
freq = 4
era_var = np.empty((days_ref, levs,)+grid_shape)
temp_var = np.empty((freq, levs,)+grid_shape)

for i in range(days_ref):
    for j in range(freq):
        file_num = freq*i+j
        with nc.Dataset(filenames[VAR][file_num], 'r') as nc_io:
            temp_var[j, ...] = nc_io.variables[nc_keys[VAR]][0, -20:, ...]
    era_var[i, ...] = f(temp_var, axis=0)
era_var = era_var - 273.15
era_var = era_var[:, ::-1, ...]

print('Processing vertical coords')
VAR = 'SFP'
era_sfp = np.empty((days_ref,)+grid_shape)
era_gamma = np.empty((days_ref,)+grid_shape)
era_height = np.empty((days_ref, levs,)+grid_shape)
era_lev = np.empty((days_ref, levs,)+grid_shape)
temp_var = np.empty((freq,)+grid_shape)
for i in range(days_ref):
    for j in range(freq):
        file_num = freq*i+j
        with nc.Dataset(filenames[VAR][file_num], 'r') as nc_io:
            temp_var[j, ...] = np.squeeze(nc_io.variables[nc_keys[VAR]][...])
    era_sfp[i, ...] = f(temp_var, axis=0)

print('Lapse rate with linear regression')    
sorter = np.array([19, 18, 17, 16, 15, 14, 
                   13, 12, 11, 10,  9,  8,  
                    7,  6,  5,  4,  3,
                    2,  1,  0])

for i in range(days_ref):
    print(i)
    for j in range(grid_shape[0]):
        for k in range(grid_shape[1]):
            temp_lev = np.array(A+B*era_sfp[i, j, k])
            temp_height = 1000*metpy.calc.pressure_to_height_std(temp_lev*units.Pa).__array__()
            temp_temp = era_var[i, :, j, k]
            era_height[i, :, j, k] = temp_height
            era_lev[i, :, j, k] = temp_lev
            ind1 = np.searchsorted(temp_height, temp_height[0]+1500)
            ind2 = np.searchsorted(temp_height, temp_height[0]+2500)
            era_gamma[i, j, k] = linear_slope(temp_height[ind1:ind2], temp_temp[ind1:ind2])

data_save = (lon_025, lat_025, era_var, era_sfp, era_gamma, era_height, era_lev)
label_save = ['lon_025', 'lat_025', 'era_var', 'era_sfp', 'era_gamma', 'era_height', 'era_lev']
du.save_hdf5(data_save, label_save, ERA_dir, 'ERA_TMEAN_GAMMA_2018_2020.hdf')




    



