import sys
import h5py
import numpy as np
from glob import glob

sys.path.insert(0, '/glade/u/home/ksha/ML_repo/utils/')
import data_utils as du
from namelist_PRISM import *

hdf_io = h5py.File(BACKUP_dir+'NCEP_FNL_2016_2018_4km.hdf', 'r')
REGRID_TMAX = hdf_io['TMAX_REGRID'][...]
REGRID_TMIN = hdf_io['TMIN_REGRID'][...]
lon = hdf_io['lon'][...]
lat = hdf_io['lat'][...]
lon_c = hdf_io['lon_c'][...]
lat_c = hdf_io['lat_c'][...]
etopo = hdf_io['etopo_US'][...]
etopo_regrid = hdf_io['etopo_regrid'][...]
hdf_io.close()

temp_data = np.load(BACKUP_dir+'TMAX_STD.npy')
FIT_BASE = temp_data[()]['FIT_MEAN'][...]
FIT_BASE_leap = temp_data[()]['FIT_MEAN_leap'][...]
FIT_TMAX = np.concatenate((FIT_BASE_leap, FIT_BASE, FIT_BASE[:-1, ...]), axis=0)
STD_BASE = temp_data[()]['FIT_STD'][...]
STD_BASE_leap = temp_data[()]['FIT_STD_leap'][...]
STD_TMAX = np.concatenate((STD_BASE_leap, STD_BASE, STD_BASE[:-1, ...]), axis=0)

temp_data = np.load(BACKUP_dir+'TMIN_STD.npy')
FIT_BASE = temp_data[()]['FIT_MEAN'][...]
FIT_BASE_leap = temp_data[()]['FIT_MEAN_leap'][...]
FIT_TMIN = np.concatenate((FIT_BASE_leap, FIT_BASE, FIT_BASE[:-1, ...]), axis=0)
STD_BASE = temp_data[()]['FIT_STD'][...]
STD_BASE_leap = temp_data[()]['FIT_STD_leap'][...]
STD_TMIN = np.concatenate((STD_BASE_leap, STD_BASE, STD_BASE[:-1, ...]), axis=0)

L = len(STD_TMAX)
grid_shape = REGRID_TMAX.shape
FIT_TMAX_interp = np.ones(grid_shape)*999
STD_TMAX_interp = np.ones(grid_shape)*999
FIT_TMIN_interp = np.ones(grid_shape)*999
STD_TMIN_interp = np.ones(grid_shape)*999

for i in range(L):
    FIT_TMAX_interp[i, ...] = du.interp2d_wraper(lon_c, lat_c, FIT_TMAX[i, ...], lon, lat)
    STD_TMAX_interp[i, ...] = du.interp2d_wraper(lon_c, lat_c, STD_TMAX[i, ...], lon, lat)
    FIT_TMIN_interp[i, ...] = du.interp2d_wraper(lon_c, lat_c, FIT_TMIN[i, ...], lon, lat)
    STD_TMIN_interp[i, ...] = du.interp2d_wraper(lon_c, lat_c, STD_TMIN[i, ...], lon, lat)



REGRID_TMAX = (REGRID_TMAX - FIT_TMAX_interp)/(STD_TMAX_interp)
REGRID_TMIN = (REGRID_TMIN - FIT_TMIN_interp)/(STD_TMIN_interp)

labels = ['FIT_TMAX', 'FIT_TMIN', 'STD_TMAX', 'STD_TMIN', 'TMAX_REGRID', 'TMIN_REGRID', 
          'lon', 'lat', 'etopo', 'etopo_regrid']
data = (FIT_TMAX_interp, FIT_TMIN_interp, STD_TMAX_interp, STD_TMIN_interp, REGRID_TMAX, REGRID_TMIN, 
        lon, lat, etopo, etopo_regrid)

du.save_hdf5(data, labels, out_dir = BACKUP_dir, filename='NCEP_std_2016_2018_4km.hdf')























