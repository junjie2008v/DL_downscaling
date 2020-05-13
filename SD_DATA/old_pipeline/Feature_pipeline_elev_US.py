
# general tools
import sys
import subprocess
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/')
import data_utils as du
import pipeline_utils as pu
from namelist import * 

# pipeline macros
del_old = True # <--------- !!! delete old batches


norm = 'norm_std'
gaps  = [2]
sizes = [96]

# clean-up
if del_old:
    cmd = 'rm {}ETOPO*npy'.format(BATCH_dir); print(cmd)
    subprocess.call(cmd, shell=True)
    cmd = 'rm {}ETOPO*npy'.format(BATCH_dir+'temp_batches/'); print(cmd)
    subprocess.call(cmd, shell=True)

# etopo fields
input_2d = {}; keys_2d = ['etopo_4km', 'etopo_regrid']
with h5py.File(PRISM_dir+'PRISM_TMAX_features_2015_2020.hdf', 'r') as hdf_io:
    for key in keys_2d:
        input_2d[key] = hdf_io[key][...]
# land mask
with h5py.File(PRISM_dir+'PRISM_TMAX_features_2015_2020.hdf', 'r') as hdf_io:
    land_mask = hdf_io['land_mask'][...]

land_mask[:ind_tune , :] = True # trans + tuning domain

# loop over sizes
for i, size in enumerate(sizes):
    gap = gaps[i]
    NAME = 'ETOPO_BATCH_{}_'.format(size)

    # random cropping + batch gen
    FEATURE = pu.random_cropping_2d(input_2d, keys_2d, land_mask, size, gap, rnd_range=1)
    FEATURE = pu.feature_norm(FEATURE, method=norm, self_norm=True)
    pu.batch_gen(FEATURE, batch_size, BATCH_dir, NAME, 0);
