

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

latlim = [24, 60]; lonlim = [-150, -100]

dx = 0.25; dy = 0.25
lon_025, lat_025 = np.meshgrid(np.arange(lonlim[0], lonlim[1]+dx, dx), np.arange(latlim[0], latlim[1]+dy, dy))
dx = 0.0416666666666; dy = 0.0416666666666
lon_4km, lat_4km = np.meshgrid(np.arange(lonlim[0], lonlim[1]+dx, dx), np.arange(latlim[0], latlim[1]+dy, dy))

import shapely
import cartopy.io.shapereader as shpreader
from cartopy.io.shapereader import Reader

shpfilename = shpreader.natural_earth(resolution='110m', category='physical', name='ocean')
land_shp = Reader(shpfilename)

shape_4km = lon_4km.shape
land_id = np.ones(shape_4km)*np.nan
for i in range(shape_4km[0]):
    for j in range(shape_4km[1]):
        temp_point = shapely.geometry.Point(lon_4km[i, j], lat_4km[i, j])
        for n, shp in enumerate(land_shp.records()):
            if shp.geometry.contains(temp_point):
                land_id[i, j] = n

land_mask = ~np.isnan(land_id)

# ETOPO interp
print('Process ETOPO')
with nc.Dataset(BACKUP_dir+'ETOPO1_Ice_g_gmt4.grd') as nc_obj:
    etopo_x = nc_obj.variables['x'][2000:] # subsetting north america
    etopo_y = nc_obj.variables['y'][2000:]
    etopo_z = nc_obj.variables['z'][2000:, 2000:]
etopo_lon, etopo_lat = np.meshgrid(etopo_x, etopo_y)

etopo_4km    = du.interp2d_wraper(etopo_lon, etopo_lat, etopo_z, lon_4km, lat_4km, method=interp_method)
etopo_025    = du.interp2d_wraper(etopo_lon, etopo_lat, etopo_z, lon_025, lat_025, method=interp_method)
etopo_regrid = du.interp2d_wraper(lon_025, lat_025, etopo_025, lon_4km, lat_4km, method=interp_method)

# save hdf
tuple_save = (lon_4km, lat_4km, etopo_4km, etopo_regrid, lon_025, lat_025, etopo_025, land_mask)
label_save = ['lon_4km', 'lat_4km', 'etopo_4km', 'etopo_regrid', 
              'lon_025', 'lat_025', 'etopo_025', 'land_mask']
du.save_hdf5(tuple_save, label_save, out_dir=PRISM_dir, filename='ETOPO_regrid.hdf')
