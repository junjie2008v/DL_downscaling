
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
del_old = False # <--------- !!! delete old batches
aug = False # <------------- !!! aug on-off 

norm = 'norm_std'
seasons = ['djf', 'mam', 'jja', 'son']
domains = ['MIX']

vars  = ['TMEAN']
gaps  = [16] # mean grid point distance between croppings (base number, will be randomly moved around)
sizes = [64]

# clean-up
if del_old:
    cmd = 'rm {}T*npy'.format(BATCH_dir); print(cmd)
    subprocess.call(cmd, shell=True)
    cmd = 'rm {}T*npy'.format(BATCH_dir+'temp_batches/'); print(cmd)
    subprocess.call(cmd, shell=True)

# train date range
# datetime info
N_days = 365 + 366 + 365 + 365 + 365 # 2015-2020 (period ending)
date_list = [datetime(2015, 1, 1, 0) + timedelta(days=x) for x in range(N_days)]
#
N_train = 365 + 366 + 365 
train_list = [datetime(2015, 1, 1, 0) + timedelta(days=x) for x in range(N_train)]
pick_train = du.dt_match(date_list, train_list)
IND_train = du.season_ind_sep(train_list, key_format='{}_train') # split by seasons
#
N_valid = 365
valid_list = [datetime(2018, 1, 1, 0) + timedelta(days=x) for x in range(N_valid)]
pick_valid = du.dt_match(date_list, valid_list)
IND_valid = du.season_ind_sep(valid_list, key_format='{}_valid') # split by seasons

# etopo fields
input_2d = {}
keys_2d = ['etopo_4km', 'etopo_regrid']
# etopo from an example
with h5py.File(PRISM_dir+'PRISM_TMAX_features_2015_2020.hdf', 'r') as hdf_io:
    for key in keys_2d:
        input_2d[key] = hdf_io[key][...]
# land mask
with h5py.File(PRISM_dir+'PRISM_TMAX_features_2015_2020.hdf', 'r') as hdf_io:
    land_mask = hdf_io['land_mask'][...]
    
# loop: var --> size --> season --> domain --> aug options
for var in vars:
    print('===== Cropping {} ====='.format(var))
    
    # import pre-processed features
    # T2 fields
    input_train_3d = {}
    input_valid_3d = {}
    keys_3d = ['{}_4km'.format(var), '{}_REGRID'.format(var), '{}_CLIM_4km'.format(var), '{}_CLIM_REGRID'.format(var)]
    with h5py.File(PRISM_dir+'PRISM_{}_features_2015_2020.hdf'.format(var), 'r') as hdf_io:
        for key in keys_3d:
            input_train_3d[key] = hdf_io[key][pick_train, ...]
            input_valid_3d[key] = hdf_io[key][pick_valid, ...]
        
    # land mask
    mask = {}
    keys_mask = ['ORI', 'MIX', 'SUB']
    # land mask gen (True = discard; False = accept)
    mask['ORI'] = np.copy(land_mask); mask['ORI'][ind_trans:, :] = True # training + tuning
    mask['MIX'] = np.copy(land_mask); mask['MIX'][:ind_tune , :] = True # transferring + tuning
    mask['SUB'] = np.copy(land_mask); mask['SUB'][:ind_trans, :] = True # transferring
    
    for i, size in enumerate(sizes):
        gap = gaps[i]
        
        for sea in seasons:
            # output names
            NAME_train = {}; NAME_valid = {}
            # indices
            ind_train = IND_train['{}_train'.format(sea)]
            ind_valid = IND_valid['{}_valid'.format(sea)]
            
            num_train = {}; num_valid = {} # aug ind (numbering augmented batches)
            for domain in domains:
                num_train[domain] = 0
                num_valid[domain] = 0
                
                # batch augmentation and gen with 90-degree rotations
                for rn in range(5):
                    # rn == 0 no rotates
                    if rn == 0:
                        # output batch filenames
                        NAME_train = '{}_BATCH_{}_T{}_{}'.format(var, size, domain, sea)
                        NAME_valid = '{}_BATCH_{}_V{}_{}'.format(var, size, domain, sea)

                        # random cropping + batch gen
                        print('----- Training data process -----')
                        FEATURE_train = pu.random_cropping(input_train_3d, keys_3d, input_2d, keys_2d, mask[domain], size, gap, ind_train, rnd_range=2)
                        FEATURE_train = pu.feature_norm(FEATURE_train, method=norm)
                        pu.batch_gen(FEATURE_train, batch_size, BATCH_dir, NAME_train, 0);
                        
                        print('----- Validation data process -----')
                        FEATURE_valid = pu.random_cropping(input_valid_3d, keys_3d, input_2d, keys_2d, mask[domain], size, gap, ind_valid, rnd_range=2)
                        FEATURE_valid = pu.feature_norm(FEATURE_valid, method=norm)
                        pu.batch_gen(FEATURE_valid, batch_size, BATCH_dir, NAME_valid, 0);
                        
                    elif aug:
                        print('Rotation round: {}'.format(rn))
                        # Feature rotations
                        for key in keys_3d:
                            input_train_3d[key] = np.rot90(input_train_3d[key], k=1, axes=(1, 2))
                            input_valid_3d[key] = np.rot90(input_valid_3d[key], k=1, axes=(1, 2))
                        for key in keys_2d:
                            input_2d[key] = np.rot90(input_2d[key], k=1)
                        # land mask rotation
                        mask[domain] = np.rot90(mask[domain], k=1)
                         
                        if rn < 4:
                            # 0 < rn < 4 aug rotates
                            NAME_train = '{}_BATCH_{}_T{}AUG_{}'.format(var, size, domain, sea)
                            NAME_valid = '{}_BATCH_{}_V{}AUG_{}'.format(var, size, domain, sea)

                            print('----- Training data process (aug) -----')
                            FEATURE_train = pu.random_cropping(input_train_3d, keys_3d, input_2d, keys_2d, mask[domain], size, gap, ind_train)
                            FEATURE_train = pu.feature_norm(FEATURE_train, method=norm)
                            ind0 = pu.batch_gen(FEATURE_train, batch_size, BATCH_dir, NAME_train, num_train[domain]);
                            num_train[domain] += ind0

                            print('----- Validation data process (aug)-----')
                            FEATURE_valid = pu.random_cropping(input_valid_3d, keys_3d, input_2d, keys_2d, mask[domain], size, gap, ind_valid)
                            FEATURE_valid = pu.feature_norm(FEATURE_valid, method=norm)
                            ind0 = pu.batch_gen(FEATURE_valid, batch_size, BATCH_dir, NAME_valid, num_valid[domain]);
                            num_valid[domain] += ind0

                        else:
                            # rn=4 rotates back to rn=0, no croppings
                            print('Rotation ends with rn = {}'.format(rn))
                            continue; 
                    else:
                        continue;
