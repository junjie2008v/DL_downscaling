'''
Downscaling pre-processing
Coverting PCIC PRISM netCDF4 files to a single hdf5
Creating lat/lon and land mask info in BC
'''

# general tools
import sys
from glob import glob
from os.path import basename
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np
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

VARS = ['PCT', 'TMAX', 'TMIN', 'TMEAN']
# filenames
files = {}
files['PCT']   = PRISM_CLIM_BC_dir+'PCT_PRISM_BC_1981_2010.nc'
files['TMAX']  = PRISM_CLIM_BC_dir+'TMAX_PRISM_BC_1981_2010.nc'
files['TMIN']  = PRISM_CLIM_BC_dir+'TMIN_PRISM_BC_1981_2010.nc'
# nc variable keys
nc_keys = {}
nc_keys['PCT'] = 'pr'
nc_keys['TMAX'] = 'tmax'
nc_keys['TMIN'] = 'tmin'

# import HR lon/lat from a single file
with nc.Dataset(files['TMAX'], 'r') as nc_obj:
    land_mask_800m = nc_obj.variables[nc_keys['TMAX']][0, ...]
    x_800m = nc_obj.variables['lon'][...]
    y_800m = nc_obj.variables['lat'][...]
    
lon_800m, lat_800m = np.meshgrid(x_800m, y_800m)

# defining HR and LR lon/lat
print('BC domain lat/lon creation')
latlim = [np.nanmin(lat_800m), np.nanmax(lat_800m)]
lonlim = [np.nanmin(lon_800m), np.nanmax(lon_800m)]
# 4km
dx_4km = 0.0416666666666
dy_4km = 0.0416666666666
lon_4km, lat_4km = np.meshgrid(np.arange(lonlim[0], lonlim[1], dx_4km), np.arange(latlim[0], latlim[1], dy_4km))
# 0.25 degree
dx_025 = 0.25
dy_025 = 0.25
lon_025, lat_025 = np.meshgrid(np.arange(lonlim[0], lonlim[1], dx_025), np.arange(latlim[0], latlim[1], dy_025))
print('\tlon_800m.shape={}\n\tlon_4km.shape={}\n\tlon_025.shape={}'.format(lon_800m.shape, lon_4km.shape, lon_025.shape))

shape_4km = lon_4km.shape
shape_025 = lon_025.shape

# BC land mask
print('Preparing BC land mask (slow)')
# griddata interp
input_points = (lon_800m.flatten(), lat_800m.flatten())
land_mask_800m = np.ma.filled(land_mask_800m, fill_value=np.nan) # converting masked elements to np.nan
land_mask = griddata(input_points, land_mask_800m.flatten(), (lon_4km, lat_4km), method='linear')
land_mask = np.isnan(land_mask)

# Watershed mask
## if re-run this script (with old files)
print('Copy BC watershed mask from the old file')
with h5py.File(PRISM_dir + 'PRISM_regrid_BC_clim.hdf', 'r') as hdf_io:
    wshed_id = hdf_io['wshed_mask'][...]

# ******************************************* #
# create from *.shp file (very slow)
# print('Preparing BC watershed mask (slow)')
# wshed_shp = Reader(fig_dir+'wshed_hires/MajorHydroWatershedsProject.shp')
# wshed_id = np.ones(shape_4km)*999
# for i in range(shape_4km[0]):
#     for j in range(shape_4km[1]):
#         temp_point = shapely.geometry.Point(lon_4km[i, j], lat_4km[i, j])
#         for n, wshed in enumerate(wshed_shp.records()):
#             if wshed.geometry.contains(temp_point):
#                 wshed_id[i, j] = n
# ******************************************* #

print('Preparing ETOPO data')
with nc.Dataset(BACKUP_dir+'ETOPO1_Ice_g_gmt4.grd') as nc_obj:
    etopo_x = nc_obj.variables['x'][2000:7000] # subsetting north america
    etopo_y = nc_obj.variables['y'][6000:]
    etopo_z = nc_obj.variables['z'][6000:, 2000:7000]
etopo_lon, etopo_lat = np.meshgrid(etopo_x, etopo_y)
# interp.
etopo_4km = du.interp2d_wraper(etopo_lon, etopo_lat, etopo_z, lon_4km, lat_4km, method=interp_method)
etopo_025 = du.interp2d_wraper(etopo_lon, etopo_lat, etopo_z, lon_025, lat_025, method=interp_method)
etopo_regrid = du.interp2d_wraper(lon_025, lat_025, etopo_025, lon_4km, lat_4km, method=interp_method)


# dictionary (tuple)
dict_4km = {}
dict_025 = {}
dict_regrid = {}

# hdf5 labels
label_4km = []
label_025 = []
label_regrid = []

# loop over variables
for var in VARS:
    print('===== Process {} ===== '.format(var))
    # load prism
    if var in ['PCT', 'TMAX', 'TMIN']:
        with nc.Dataset(files[var], 'r') as nc_obj:
            prism = nc_obj[nc_keys[var]][...]
        # clean up _fillvals
        prism = np.ma.filled(prism, fill_value=np.nan)
        prism[np.abs(prism) > 999] = np.nan
    else:
        # TMEAN = 0.5*TMAX + 0.5*TMIN
        with nc.Dataset(files['TMAX'], 'r') as nc_obj:
            prism1 = nc_obj[nc_keys['TMAX']][...]
        with nc.Dataset(files['TMIN'], 'r') as nc_obj:
            prism2 = nc_obj[nc_keys['TMIN']][...]
        # clean up _fillvals
        prism1 = np.ma.filled(prism1, fill_value=np.nan)
        prism2 = np.ma.filled(prism2, fill_value=np.nan)
        prism1[np.abs(prism1) > 999] = np.nan
        prism2[np.abs(prism2) > 999] = np.nan
        prism = 0.5*prism1 + 0.5*prism2
        
    prism_800m = prism[:-1, ...] # <-- start with 800m
    prism_4km = np.empty((12,)+shape_4km)
    prism_025 = np.empty((12,)+shape_025)
    prism_regrid = np.empty(prism_4km.shape)
    
    for i in range(12):
        # re-griding
        temp_4km = du.interp2d_wraper(lon_800m, lat_800m, prism_800m[i, ...], lon_4km, lat_4km, method=interp_method)
        temp_4km[land_mask] = np.nan
        temp_025 = du.interp2d_wraper(lon_800m, lat_800m, prism_800m[i, ...], lon_025, lat_025, method=interp_method)
        temp_regrid = du.interp2d_wraper(lon_025, lat_025, temp_025, lon_4km, lat_4km, method=interp_method)
        temp_regrid[land_mask] = np.nan
        # allocation
        prism_4km[i, ...] = temp_4km
        prism_025[i, ...] = temp_025
        prism_regrid[i, ...] = temp_regrid
        
    # collecting fields
    dict_4km[var] = prism_4km    
    dict_025[var] = prism_025
    dict_regrid[var] = prism_regrid
    # collecting label
    label_4km.append(var+'_4km')
    label_025.append(var+'_025')
    label_regrid.append(var+'_REGRID')
    
# dictionary to tuple
tuple_4km = tuple(dict_4km.values())
tuple_025 = tuple(dict_025.values())
tuple_regrid = tuple(dict_regrid.values())
tuple_etopo = (etopo_4km, etopo_025, etopo_regrid)
tuple_grids = (lon_025, lat_025, lon_4km, lat_4km, land_mask, wshed_id)

# mark labels
label_etopo = ['etopo_4km', 'etopo_025', 'etopo_regrid']
label_grids = ['lon_025', 'lat_025', 'lon_4km', 'lat_4km', 'land_mask', 'wshed_mask']

# save hdf
tuple_save = tuple_4km + tuple_025 + tuple_regrid + tuple_etopo + tuple_grids
label_save = label_4km + label_025 + label_regrid + label_etopo + label_grids
du.save_hdf5(tuple_save, label_save, out_dir=PRISM_dir, filename='PRISM_regrid_BC_clim.hdf')

