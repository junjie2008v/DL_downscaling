import sys
from os.path import basename
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np
#from scipy.stats import linregress
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
# access original lat, lon
with nc.Dataset(JRA_TAIR_dir + 'anl_mdl.011_tmp.reg_tl319.2019122100_2019123118.sha416570.nc', 'r') as nc_io:
    x025 = nc_io['g4_lon_3'][...]-360
    y025 = nc_io['g4_lat_2'][...]
lon_025, lat_025 = np.meshgrid(x025, y025)

# nc variable keys
nc_keys = {}
nc_keys['TMEAN'] = 'TMP_GDS4_HYBL'
nc_keys['SFP'] = 'PRES_GDS4_SFC'
# filenames
filenames = {}
filenames['TMEAN'] = sorted(glob(JRA_TAIR_dir+'*.nc'))
filenames['SFP'] = sorted(glob(JRA_SFP_dir+'*.nc'))

# aggregation method
method = 'nanmean'
# datetime info
days_ref = 365 + 365 # 2018 - 2019-08-31

# array size
levs = 20 # veritial levels
freq = 4 # 4 times a day
grid_shape = lon_025.shape

mon_days_365 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
mon_days_366 = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

print('Processing air temp')
VAR = 'TMEAN'
f = eval(method)
jra_var = np.empty((days_ref, levs,)+grid_shape)

count = 0
for i, name in enumerate(filenames[VAR]):
    start_date = datetime.strptime(basename(name)[26:36], '%Y%m%d%H')
    end_date = datetime.strptime(basename(name)[37:47], '%Y%m%d%H')
    L_day = (end_date-start_date).days+1
    with nc.Dataset(name, 'r') as nc_io:
        T2 = np.squeeze(nc_io.variables[nc_keys[VAR]][...])
    T2_fold = T2.reshape((L_day, 4)+(levs,)+grid_shape)
    TMEAN_temp = f(T2_fold, axis=1)
    jra_var[count:count+L_day, ...] = TMEAN_temp
    count += L_day
jra_var = jra_var - 273.15
        
print('Processing vertical coords')
VAR = 'SFP'
jra_sfp = np.empty((days_ref,)+grid_shape)
jra_base = np.empty((days_ref,)+grid_shape)
jra_gamma = np.empty((days_ref,)+grid_shape)
jra_height = np.empty((days_ref, levs,)+grid_shape)
jra_lev = np.empty((days_ref, levs,)+grid_shape)
count = 0
for i, name in enumerate(filenames[VAR]):
    start_date = datetime.strptime(basename(name)[28:38], '%Y%m%d%H')
    start_year = start_date.year
    start_mon = start_date.month
    if start_year == 2016:
        L_day = mon_days_366[start_mon-1]
    else:
        L_day = mon_days_365[start_mon-1]
    with nc.Dataset(name, 'r') as nc_io:
        P = np.squeeze(nc_io.variables[nc_keys[VAR]][...])

    P_fold = P.reshape((L_day, 4)+grid_shape)
    P_temp = f(P_fold, axis=1)
    jra_sfp[count:count+L_day, ...] = P_temp
    count += L_day

print('Lapse rate with linear regression')    
#
for i in range(days_ref):
    print(i)
    for j in range(grid_shape[0]):
        for k in range(grid_shape[1]):
            temp_lev, temp_height = vu.hybrid_to_m(slp=jra_sfp[i, j, k])
            temp_temp = jra_var[i, :, j, k]
            jra_height[i, :, j, k] = temp_height
            jra_lev[i, :, j, k] = temp_lev
            ind1 = np.searchsorted(temp_height, temp_height[0]+1500)
            ind2 = np.searchsorted(temp_height, temp_height[0]+2500)
            jra_gamma[i, j, k] = linear_slope(temp_height[ind1:ind2], temp_temp[ind1:ind2])
            # local air zone
#             if temp_lev[0] >= 92500:
#                 ind_925 = np.searchsorted(temp_lev, 92500, sorter=sorter)
#                 ind_925 = sorter[ind_925]
#                 ind_850 = np.searchsorted(temp_lev, 85000, sorter=sorter)
#                 ind_850 = sorter[ind_850]
#                 jra_gamma[i, j, k] = linear_slope(temp_height[ind_925:ind_850], temp_temp[ind_925:ind_850])
#                 jra_base[i, j, k] = np.nan
#             elif temp_lev[0] >= 85000:
#                 ind_850 = np.searchsorted(temp_lev, 85000, sorter=sorter)
#                 ind_850 = sorter[ind_850]
#                 ind_700 = np.searchsorted(temp_lev, 70000, sorter=sorter)
#                 ind_700 = sorter[ind_700]
#                 jra_gamma[i, j, k] = linear_slope(temp_height[ind_850:ind_700], temp_temp[ind_850:ind_700])
#                 jra_base[i, j, k] = np.nan
#             # free air zone
#             else:
#                 jra_base[i, j, k] = temp_temp[0] # use air temp as ground ref
#                 ind_free = np.searchsorted(temp_lev, temp_lev[0]-15000, sorter=sorter)
#                 ind_free = sorter[ind_free]
#                 jra_gamma[i, j, k] = linear_slope(temp_height[0:ind_free], temp_temp[0:ind_free])

data_save = (lon_025, lat_025, jra_var, jra_sfp, jra_gamma, jra_height, jra_lev)
label_save = ['lon_025', 'lat_025', 'jra_var', 'jra_sfp', 'jra_gamma', 'jra_height', 'jra_lev']
du.save_hdf5(data_save, label_save, JRA_dir, 'JRA_TMEAN_GAMMA_2018_2020.hdf')




    



