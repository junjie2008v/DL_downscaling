'''
Downscaling pre-processing
Coverting PRISM original files to a single hdf5
one month one frame
'''

# general tools
import sys
from glob import glob
from os.path import basename
from datetime import datetime, timedelta

# data tools
import h5py
import rasterio
import numpy as np

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/')
import data_utils as du
from namelist import * 

# PRISM original grid spacing
dx = 0.0416666666666; dy = 0.0416666666666

vars = ['PCT', 'TMAX', 'TMIN', 'TMEAN']
dirs = {}
dirs['PCT']   = PRISM_CLIM_US_dir+'PCT/'
dirs['TMAX']  = PRISM_CLIM_US_dir+'TMAX/'
dirs['TMIN']  = PRISM_CLIM_US_dir+'TMIN/'
dirs['TMEAN'] = PRISM_CLIM_US_dir+'TMEAN/'

for var in vars:
    print('===== Extracting {} ====='.format(var))
     # get raw file names
    file_dir = sorted(glob(dirs[var]+'*_extract/'))
    print('Number of raws: {}'.format(len(file_dir)))
    
    # get geo info from a single example
    temp_dir = file_dir[0]
    temp_file = glob(temp_dir+'*.bil')[0]
    temp_io = rasterio.open(temp_file, 'r')
    bounds = list(temp_io.bounds)
    temp_data = np.squeeze(temp_io.read())
    data_mask = np.squeeze(temp_io.dataset_mask())
    N_lon = temp_io.width
    N_lat = temp_io.height
    temp_io.close()
    lon, lat = np.meshgrid(np.arange(bounds[0], bounds[0]+dx*N_lon, dx), np.arange(bounds[1], bounds[1]+dy*N_lat, dy))
    
    # loop over files
    PRISM_PCT = np.empty((len(file_dir), N_lat, N_lon))
    for i, temp_dir in enumerate(file_dir):
        temp_file = glob(temp_dir+'*.bil')[0]
        #print(temp_file)
        temp_io = rasterio.open(temp_file, 'r')
        temp_data = np.squeeze(temp_io.read())
        temp_io.close()
        # vals
        temp_data[data_mask==0]=np.nan
        temp_data = np.flipud(temp_data)
        PRISM_PCT[i, ...] = temp_data

    # save hdf5
    tuple_save = (PRISM_PCT, lon, lat)
    label_save = ['PRISM_{}'.format(var), 'lon', 'lat']
    du.save_hdf5(tuple_save, label_save, out_dir=PRISM_dir, filename='PRISM_{}_clim.hdf'.format(var))