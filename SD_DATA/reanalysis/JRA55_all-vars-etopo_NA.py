import sys
from os.path import basename
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np
from numpy import nansum, nanmax, nanmin, nanmean
import netCDF4 as nc

# geo tools
import shapely
from scipy.interpolate import griddata
from cartopy.io.shapereader import Reader

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/')
import data_utils as du
from namelist import * 

print('Import lat/lon and land mask')
with h5py.File(PRISM_dir+'land_mask_NA.hdf', 'r') as hdf_io:
    lon_4km = hdf_io['lon_4km'][...]
    lat_4km = hdf_io['lat_4km'][...]
    lon_025_clean = hdf_io['lon_025'][...]
    lat_025_clean = hdf_io['lat_025'][...]
#     etopo_regrid_clean = hdf_io['etopo_regrid'][...]
    land_mask = hdf_io['land_mask'][...]
    
print('Raw netCDF4 process')
# JRA 55 is stored as one month per nc
#     data available every 6-hour (e.g., 31*4)
# Vars:
#     T2: TMP_GDS4_HTGL
#     lon: g4_lon_2
#     lat: g4_lat_1
# Time:
#     2015-01-01 to 2019-12-31
# data has been subsetted to NA (matches with land mask)

# access original lat, lon
with nc.Dataset(JRA_TMEAN_dir+'anl_surf.011_tmp.reg_tl319.2015010100_2015013118.sha410426.nc', 'r') as nc_io:
    x025 = nc_io['g4_lon_2'][...]-360 # <--- fix to [-180, 180]
    y025 = nc_io['g4_lat_1'][...]
    
y025 = np.flipud(y025)
lon_025, lat_025 = np.meshgrid(x025, y025)

print('Preparing ETOPO data')
with nc.Dataset(BACKUP_dir+'ETOPO1_Ice_g_gmt4.grd') as nc_obj:
    etopo_x = nc_obj.variables['x'][2000:] # subsetting north america
    etopo_y = nc_obj.variables['y'][6000:]
    etopo_z = nc_obj.variables['z'][6000:, 2000:]
etopo_lon, etopo_lat = np.meshgrid(etopo_x, etopo_y)
# interp.
etopo_025 = du.interp2d_wraper(etopo_lon, etopo_lat, etopo_z, lon_025, lat_025, method=interp_method)
etopo_regrid = du.interp2d_wraper(lon_025, lat_025, etopo_025, lon_4km, lat_4km, method=interp_method)

# processing keywords
VARS = ['TMEAN']
# nc variable keys
nc_keys = {}
nc_keys['TMEAN'] = 'TMP_GDS4_HTGL' # analysis
# filenames
filenames = {}
filenames['TMEAN'] = sorted(glob(JRA_TMEAN_dir+'*.nc'))
# aggregation method
method = {}
method['TMEAN'] = 'nanmean'


# datetime info
days_ref = 365 + 366 + 365 + 365 + 365 # 2016 - 2020

# !!! <---- assuming no missing date
# datetime reference (in the case of missing date)
#base = datetime(2015, 1, 1, 0)
#date_list = [base + timedelta(days=x) for x in range(days_ref)]
#print('last day: {}'.format(date_list[-1]))

mon_days_365 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
mon_days_366 = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# allocation
grid_shape = lon_025.shape
temp_var = np.empty((4,)+grid_shape)
jra_var = np.empty((days_ref,)+grid_shape)
jra_interp = np.empty((days_ref,)+lon_4km.shape)
jra_025 = np.empty((days_ref,)+lon_025_clean.shape)

for VAR in VARS:
    print('Process {}'.format(VAR))
    count = 0
    f = eval(method[VAR])
    for i, name in enumerate(filenames[VAR]):
        start_date = datetime.strptime(basename(name)[27:37], '%Y%m%d%H')
        start_year = start_date.year
        start_mon = start_date.month
        if start_year == 2016:
            L_day = mon_days_366[start_mon-1]
        else:
            L_day = mon_days_365[start_mon-1]
        
        # import vars
        with nc.Dataset(name, 'r') as nc_io:
            T2 = nc_io.variables[nc_keys[VAR]][...] # (time, lat, lon)
            
        T2_fold = T2.reshape((L_day, 4)+grid_shape)
        TMEAN_temp = f(T2_fold, axis=1)
        
        for d in range(L_day):
            jra_var[count+d:count+d+1, ...] = np.flipud(TMEAN_temp[d, ...])
        count += L_day
    
    print('Interpolation')
    for i in range(days_ref):
        if i%200 == 0:
            print('\tday index: {}'.format(i))
        jra_025[i, ...] = du.interp2d_wraper(lon_025, lat_025, jra_var[i, ...], 
                                             lon_025_clean, lat_025_clean, method=interp_method)
        temp_interp = du.interp2d_wraper(lon_025, lat_025, jra_var[i, ...], lon_4km, lat_4km, method=interp_method)
        # land mask applied
        temp_interp[land_mask] = np.nan
        jra_interp[i, ...] = temp_interp
        
    print('Feature engineering')
    if VAR in ['TMAX', 'TMIN', 'TMEAN']:
        print('\tK to C')
        jra_025 = jra_025 - 273.15
        jra_var = jra_var - 273.15
        jra_interp = jra_interp - 273.15
        
    data_save = (lon_025, lat_025, lon_4km, lat_4km, jra_var, jra_025, jra_interp, etopo_025, land_mask)
    label_save = ['lon_raw', 'lat_raw', 'lon_4km', 'lat_4km', 'jra_raw', 'jra_025', '{}_REGRID'.format(VAR), 'etopo_raw',  'land_mask']
    du.save_hdf5(data_save, label_save, JRA_dir, 'JRA_{}_features_2015_2020.hdf'.format(VAR))
