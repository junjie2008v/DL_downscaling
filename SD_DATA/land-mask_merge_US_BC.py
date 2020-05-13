import sys
from glob import glob

# data tools
import h5py
import imageio
import netCDF4 as nc
import numpy as np
from scipy.interpolate import griddata

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/')
import data_utils as du
from namelist import * 

# ====== If the land mask file does not exist ===== #
# # import BC and US land mask
# with h5py.File(PRISM_dir+'PRISM_regrid_BC_clim.hdf', 'r') as h_io:
#     land_mask_BC = h_io['land_mask'][...]
#     lon_BC = h_io['lon_4km'][...]
#     lat_BC = h_io['lat_4km'][...]

# with h5py.File(PRISM_dir+'PRISM_regrid_clim.hdf', 'r') as h_io:
#     land_mask_US = h_io['land_mask'][...]
#     lon_US = h_io['lon_4km'][...]
#     lat_US = h_io['lat_4km'][...]

# # create lat/lon reference
# lonlim = [lon_BC.min(), lon_US.max()]
# latlim = [lat_US.min(), lat_BC.max()]
# dx_4km = 0.0416666666666
# dy_4km = 0.0416666666666
# lon_4km, lat_4km = np.meshgrid(np.arange(lonlim[0], lonlim[1]+dx_4km, dx_4km), np.arange(latlim[0], latlim[1]+dy_4km, dy_4km))

# lonlim_025 = [-140, -100]
# latlim_025 = [24, 62]
# dx_025 = 0.25
# dy_025 = 0.25
# lon_025, lat_025 = np.meshgrid(np.arange(lonlim_025[0], lonlim_025[1]+dx_025, dx_025), 
#                                np.arange(latlim_025[0], latlim_025[1]+dy_025, dy_025))

# print('Preparing ETOPO data')
# with nc.Dataset(BACKUP_dir+'ETOPO1_Ice_g_gmt4.grd') as nc_obj:
#     etopo_x = nc_obj.variables['x'][2000:] # subsetting north america
#     etopo_y = nc_obj.variables['y'][6000:]
#     etopo_z = nc_obj.variables['z'][6000:, 2000:]
# etopo_lon, etopo_lat = np.meshgrid(etopo_x, etopo_y)
# # interp.
# etopo_4km = du.interp2d_wraper(etopo_lon, etopo_lat, etopo_z, lon_4km, lat_4km, method=interp_method)
# etopo_025 = du.interp2d_wraper(etopo_lon, etopo_lat, etopo_z, lon_025, lat_025, method=interp_method)
# etopo_regrid = du.interp2d_wraper(lon_025, lat_025, etopo_025, lon_4km, lat_4km, method=interp_method)

# # land mask interpolation
# print('Preparing land mask')
# new_mask_BC = griddata((lon_BC.ravel(), lat_BC.ravel()), land_mask_BC.ravel(), (lon_4km, lat_4km), method='linear')
# new_mask_US = griddata((lon_US.ravel(), lat_US.ravel()), land_mask_US.ravel(), (lon_4km, lat_4km), method='linear')
# land_mask = np.ones(lon_4km.shape)
# land_mask[new_mask_BC==0]=0
# land_mask[new_mask_US==0]=0
# land_mask = land_mask > 0
# ================================================= #

# # ====== If the land mask file exist ===== #
# with h5py.File(PRISM_dir + 'land_mask_NA.hdf', 'r') as hdf_io:
#     lon_025 = hdf_io['lon_025'][...]
#     lat_025 = hdf_io['lat_025'][...]  
#     lon_4km = hdf_io['lon_4km'][...]
#     lat_4km = hdf_io['lat_4km'][...]
#     land_mask = hdf_io['land_mask'][...]
#     etopo_4km = hdf_io['etopo_4km'][...]
#     etopo_regrid = hdf_io['etopo_regrid'][...]
# # ======================================== #
# print('land_mask_025 processing')
# flag_mask = griddata((lon_4km.ravel(), lat_4km.ravel()), land_mask.ravel(), (lon_025, lat_025), method='linear')
# land_mask_025 = flag_mask > 0

# print('land_mask_terrain processing')
# filename = '/glade/u/home/ksha/land_mask_terrain.png'
# land_mask_ct = imageio.imread(filename)
# land_mask_ct = land_mask_ct[..., 2]
# land_mask_ct = np.flipud(land_mask_ct)
# land_mask_ct = land_mask_ct.astype(np.float)
# land_mask_ct = land_mask_ct==255

# grid_shape = land_mask_ct.shape
# x_temp = np.linspace(lon_4km.min(), lon_4km.max(), grid_shape[1])
# y_temp = np.linspace(lat_4km.min(), lat_4km.max(), grid_shape[0])
# lon_temp, lat_temp = np.meshgrid(x_temp, y_temp)

# flag_mask = griddata((lon_temp.ravel(), lat_temp.ravel()), land_mask_ct.ravel(), (lon_4km, lat_4km), method='linear')
# land_mask_terrain_4km = flag_mask > 0

# print('land_mask_terrain_025 processing')
# flag_mask = griddata((lon_4km.ravel(), lat_4km.ravel()), land_mask_terrain_4km.ravel(), (lon_025, lat_025), method='linear')
# land_mask_terrain_025 = flag_mask > 0

# =======
with h5py.File(PRISM_dir + 'land_mask_NA.hdf', 'r') as hdf_io:
    lon_025 = hdf_io['lon_025'][...]
    lat_025 = hdf_io['lat_025'][...]  
    lon_4km = hdf_io['lon_4km'][...]
    lat_4km = hdf_io['lat_4km'][...]
    
    land_mask_025 = hdf_io['land_mask_025'][...]
    land_mask_terrain_025 = hdf_io['land_mask_terrain_025'][...]
    
    land_mask = hdf_io['land_mask'][...]
    land_mask_terrain_4km = hdf_io['land_mask_terrain'][...]
    
    etopo_4km = hdf_io['etopo_4km'][...]
    etopo_regrid = hdf_io['etopo_regrid'][...]

land_mask_025[:, 155:] = True
land_mask_025[152:, :] = True
land_mask_025[:5, :] = True
    
land_mask_terrain_025[:, 155:] = True
land_mask_terrain_025[145:, :] = True
land_mask_terrain_025[:5, :] = True

# save
tuple_save = (lon_4km, lat_4km, lon_025, lat_025, etopo_4km, etopo_regrid, 
              land_mask, land_mask_025, land_mask_terrain_4km, land_mask_terrain_025)
label_save = ['lon_4km', 'lat_4km',  'lon_025', 'lat_025', 'etopo_4km', 'etopo_regrid', 
              'land_mask', 'land_mask_025', 'land_mask_terrain', 'land_mask_terrain_025']
du.save_hdf5(tuple_save, label_save, out_dir=PRISM_dir, filename='land_mask_NA.hdf')

