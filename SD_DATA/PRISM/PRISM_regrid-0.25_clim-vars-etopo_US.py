
'''
Re-griding PRISM climatology by month 
(Downscaling pre-rpocessing but not used anymone)
'''
# general tools
import sys
from glob import glob

# data tools
import h5py
import numpy as np
import netCDF4 as nc

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/')
import data_utils as du
from namelist import * 

# macros
# interp_method = 'cubic'
VAR_list = ['PCT', 'TMIN', 'TMAX', 'TMEAN']

# import HR lon/lat from a single file
with h5py.File(PRISM_dir+'PRISM_PCT_2015_2020.hdf', 'r') as hdf_io:
    land_mask = hdf_io['PRISM_PCT'][0, subset_ind[0]:subset_ind[1], subset_ind[2]:subset_ind[3]]
    lon_4km = hdf_io['lon'][subset_ind[0]:subset_ind[1], subset_ind[2]:subset_ind[3]]
    lat_4km = hdf_io['lat'][subset_ind[0]:subset_ind[1], subset_ind[2]:subset_ind[3]]
land_mask = np.isnan(land_mask)

# defining LR lon/lat
dx = 0.25; dy = 0.25
latlim = [24, 49]; lonlim = [-125, -100.25]
lon_025, lat_025 = np.meshgrid(np.arange(lonlim[0], lonlim[1], dx), np.arange(latlim[0], latlim[1], dy))
print('lon_4km.shape:{}; lon_025.shape:{}'.format(lon_4km.shape, lon_025.shape))

# ETOPO interp
print('Process ETOPO')
with nc.Dataset(BACKUP_dir+'ETOPO1_Ice_g_gmt4.grd') as nc_obj:
    etopo_x = nc_obj.variables['x'][2000:7000] # subsetting north america
    etopo_y = nc_obj.variables['y'][6000:]
    etopo_z = nc_obj.variables['z'][6000:, 2000:7000]
etopo_lon, etopo_lat = np.meshgrid(etopo_x, etopo_y)

# coarse-graining ETOPO1
etopo_4km    = du.interp2d_wraper(etopo_lon, etopo_lat, etopo_z, lon_4km, lat_4km, method=interp_method)
etopo_025    = du.interp2d_wraper(etopo_lon, etopo_lat, etopo_z, lon_025, lat_025, method=interp_method)
etopo_regrid = du.interp2d_wraper(lon_025, lat_025, etopo_025, lon_4km, lat_4km, method=interp_method)
# =========================== #

# dictionary (tuple)
dict_4km = {}
dict_025 = {}
dict_regrid = {}

# hdf5 labels
label_4km = []
label_025 = []
label_regrid = []

for VAR in VAR_list:
    print('===== Process {} ===== '.format(VAR))
    
    # load prism
    with h5py.File(PRISM_dir+'PRISM_{}_clim.hdf'.format(VAR), 'r') as hdf_io:
        prism = hdf_io['PRISM_{}'.format(VAR)][...]
        
    prism_4km = prism[:, subset_ind[0]:subset_ind[1], subset_ind[2]:subset_ind[3]]
    prism_025 = np.empty((12,)+lon_025.shape)
    prism_regrid = np.empty(prism_4km.shape) #
    
    for i in range(12):

        temp_025 = du.interp2d_wraper(lon_4km, lat_4km, prism_4km[i, ...], lon_025, lat_025, method=interp_method)
        temp_regrid = du.interp2d_wraper(lon_025, lat_025, temp_025, lon_4km, lat_4km, method=interp_method)
        temp_regrid[land_mask] = np.nan
        # allocation
        prism_025[i, ...] = temp_025
        prism_regrid[i, ...] = temp_regrid
        
    # collecting fields
    dict_4km[VAR] = prism_4km    
    dict_025[VAR] = prism_025
    dict_regrid[VAR] = prism_regrid
    # collecting label
    label_4km.append(VAR+'_4km')
    label_025.append(VAR+'_025')
    label_regrid.append(VAR+'_REGRID')
    
# dictionary to tuple
tuple_4km = tuple(dict_4km.values())
tuple_025 = tuple(dict_025.values())
tuple_regrid = tuple(dict_regrid.values())
tuple_etopo = (etopo_4km, etopo_025, etopo_regrid)
tuple_grids = (lon_025, lat_025, lon_4km, lat_4km, land_mask)

# mark labels
label_etopo = ['etopo_4km', 'etopo_025', 'etopo_regrid']
label_grids = ['lon_025', 'lat_025', 'lon_4km', 'lat_4km', 'land_mask']

# save hdf
tuple_save = tuple_4km + tuple_025 + tuple_regrid + tuple_etopo + tuple_grids
label_save = label_4km + label_025 + label_regrid + label_etopo + label_grids
du.save_hdf5(tuple_save, label_save, out_dir=PRISM_dir, filename='PRISM_regrid_clim.hdf')






















