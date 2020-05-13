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
with h5py.File(PRISM_dir + 'land_mask_NA.hdf', 'r') as hdf_io:
    lon_4km = hdf_io['lon_4km'][...]
    lat_4km = hdf_io['lat_4km'][...]
    etopo_4km = hdf_io['etopo_4km'][...]
    land_mask = hdf_io['land_mask'][...]

with h5py.File(JRA_dir + 'JRA_TMEAN_GAMMA_2018_2020.hdf', 'r') as hdf_io:
    lon_025 = hdf_io['lon_025'][...]
    lat_025 = hdf_io['lat_025'][...]
    jra_gamma = hdf_io['jra_gamma'][...]

with h5py.File(JRA_dir + 'JRA_TMEAN_features_2015_2020.hdf', 'r') as hdf_io:
    TMEAN_REGRID = hdf_io['TMEAN_REGRID'][...]

with nc.Dataset(BACKUP_dir+'ETOPO1_Ice_g_gmt4.grd') as nc_obj:
    etopo_x = nc_obj.variables['x'][2000:] # subsetting north america
    etopo_y = nc_obj.variables['y'][6000:]
    etopo_z = nc_obj.variables['z'][6000:, 2000:]
etopo_lon, etopo_lat = np.meshgrid(etopo_x, etopo_y)

etopo_025 = du.interp2d_wraper(etopo_lon, etopo_lat, etopo_z, lon_025, lat_025, method=interp_method)
etopo_regrid = du.interp2d_wraper(lon_025, lat_025, etopo_025, lon_4km, lat_4km, method=interp_method)

print('Lapse rate correction')
date_ref = 365 + 365
date_list = [datetime(2018, 1, 1, 0) + timedelta(days=x) for x in range(date_ref)]
gamma_mon = [-4.4, -5.9, -7.1, -7.8, -8.1, -8.2, -8.1, -8.1, -7.7, -6.8, -5.5, -4.7]

TMEAN_fix = np.zeros((date_ref,)+lon_4km.shape)
TMEAN_correct = np.zeros((date_ref,)+lon_4km.shape)
delta_etopo = etopo_4km - etopo_regrid

for i, date in enumerate(date_list):
    mon_id = date.month-1
    TMEAN_fix[i, ...] = TMEAN_REGRID[i]+gamma_mon[mon_id]*1e-3*delta_etopo
    gamma_interp = 0.5*du.interp2d_wraper(lon_025, lat_025, jra_gamma[i, ...], lon_4km, lat_4km, method=interp_method)
    TMEAN_correct[i, ...] = TMEAN_REGRID[i]+gamma_interp*delta_etopo

TMEAN_correct[:, land_mask] = np.nan

data_save = (lon_4km, lat_4km, TMEAN_correct, TMEAN_fix, land_mask)
label_save = ['lon_4km', 'lat_4km', 'TMEAN_correct', 'TMEAN_fix', 'land_mask']
du.save_hdf5(data_save, label_save, JRA_dir, 'JRA_TMEAN_correct_2018_2020.hdf')
