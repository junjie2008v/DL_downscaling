# general tools
import sys
from copy import deepcopy
from glob import glob
from collections import Counter
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np

# deep learning tools
from tensorflow import keras

# stats tools
import scipy.optimize

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
from data_utils import *

def sinfunc_annual(t, A, p, c):
    '''
    T2 feature engineering function
    '''
    return A * np.sin(t + p) + c

def fit_sin_annual(t, y):
    '''
    OLS fit of the feature engineering function
    '''
    # initial guess
    amp0 = np.std(y) * 2.**0.5
    xshift0 = 0
    offset0 = np.mean(y)
    guess = np.array([amp0, xshift0, offset0])
    
    # OLS fit
    param, _ = scipy.optimize.curve_fit(sinfunc_annual, t, y, p0=guess, maxfev=100000000)
    return param

def random_cropping(input_3d, keys_3d, input_2d, keys_2d, land_mask, size, gap, pick_ind, ocean_f=0.33, sparse_f=0.33, rnd_range=4, clim=True):
    
    '''
    Combines multiple features in one array
    Performs random croppings
        - size: cropping size
        - gap: base grid point distance between neghbouring croppings
        - ocean_f: maximum allowance of ocean grid points divided by cropping size
        - sparse_f: maximum allowance of zero/nan grid points divided by cropping size
        - rnd_range: (conceptually) level of randomness on the cropping locations (must lower than gap)
    '''
    # var info
    var = keys_3d[0][:-4]
    
    # copy dicts before mask out domains
    input_z = deepcopy(input_2d)
    input_t = {}
    for key in keys_3d:
        input_t[key] = input_3d[key][pick_ind, ...]
    input_t = deepcopy(input_t) # copied after ind subset
    
    for key in keys_3d:
        input_t[key][:, land_mask] = np.nan
    for key in keys_2d:
        input_z[key][land_mask] = np.nan
        
    # cropping detection
    Lx, Ly = land_mask.shape # domain size
    Nx = (Lx-size+1)//gap+1  # derive number of cropping by domain size
    Ny = (Ly-size+1)//gap+1
    start_flag = np.zeros([Nx, Ny]).astype(bool) # search the bottom left corner of croppings
    # check if croppings can match "ocean_f"
    for i in range(Nx):
        for j in range(Ny):
            N_ocean_grid = np.sum(land_mask[gap*i:gap*i+size, gap*j:gap*j+size])
            if N_ocean_grid>=0 and N_ocean_grid<=ocean_f*size*size:
                start_flag[i, j] = True
            else:
                start_flag[i, j] = False
                
    # crop inputs
    L = len(pick_ind)
    N_single = np.sum(start_flag) # largest possible samples in one day
    N_total = N_single*L          # largest possible samples in all days
    CROPPINGs = np.empty((N_total, size, size, len(keys_3d)+len(keys_2d)))
    count = 0 # exact number of samples
    for i in range(L):
        for indx in range(Nx):
            for indy in range(Ny):
                if start_flag[indx, indy]:
                    ind_xs = gap*indx; ind_xe = gap*indx+size
                    ind_ys = gap*indy; ind_ye = gap*indy+size
                    # adding random shifts <--- cannot larger than "gap"
                    if ind_xs > rnd_range and ind_xe < Lx-rnd_range:
                        d = np.random.randint(-1*rnd_range, rnd_range, dtype='int')
                        ind_xs += d; ind_xe += d
                    if ind_ys > rnd_range and ind_ye < Ly-rnd_range:
                        d = np.random.randint(-1*rnd_range, rnd_range, dtype='int')
                        ind_ys += d; ind_ye += d
                    # sparsity check 
                    test_frame = input_t['{}_REGRID'.format(var)][i, ind_xs:ind_xe, ind_ys:ind_ye]
                    N_zeros = np.sum(np.logical_or(np.isnan(test_frame), test_frame==0))
                    if N_zeros < sparse_f*size*size:
                        # selected input features
                        if clim:
                            CROPPINGs[count, ..., 0] = input_t['{}_4km'.format(var)][i, ind_xs:ind_xe, ind_ys:ind_ye]
                            CROPPINGs[count, ..., 1] = input_t['{}_REGRID'.format(var)][i, ind_xs:ind_xe, ind_ys:ind_ye]
                            CROPPINGs[count, ..., 2] = input_t['{}_CLIM_4km'.format(var)][i, ind_xs:ind_xe, ind_ys:ind_ye]
                            CROPPINGs[count, ..., 3] = input_t['{}_CLIM_REGRID'.format(var)][i, ind_xs:ind_xe, ind_ys:ind_ye]
                            CROPPINGs[count, ..., 4] = input_z['etopo_4km'][ind_xs:ind_xe, ind_ys:ind_ye]
                            CROPPINGs[count, ..., 5] = input_z['etopo_regrid'][ind_xs:ind_xe, ind_ys:ind_ye]
                        else:
                            CROPPINGs[count, ..., 0] = input_t['{}_4km'.format(var)][i, ind_xs:ind_xe, ind_ys:ind_ye]
                            CROPPINGs[count, ..., 1] = input_t['{}_REGRID'.format(var)][i, ind_xs:ind_xe, ind_ys:ind_ye]
                            CROPPINGs[count, ..., 2] = input_z['etopo_4km'][ind_xs:ind_xe, ind_ys:ind_ye]
                            CROPPINGs[count, ..., 3] = input_z['etopo_regrid'][ind_xs:ind_xe, ind_ys:ind_ye]
                        count += 1
         
    print('\tNumber of croppings: {}'.format(count))
    return CROPPINGs[:count, ...]

def random_cropping_regrid(input_3d, keys_3d, input_2d, keys_2d, land_mask, size, gap, pick_ind, var='TMEAN', ocean_f=0.33, sparse_f=0.33, rnd_range=4):
    '''
    Combines multiple features in one array
    Performs random croppings
        - size: cropping size
        - gap: base grid point distance between neghbouring croppings
        - ocean_f: maximum allowance of ocean grid points divided by cropping size
        - sparse_f: maximum allowance of zero/nan grid points divided by cropping size
        - rnd_range: (conceptually) level of randomness on the cropping locations (must lower than gap)
    '''
    # copy dicts before mask out domains
    input_z = deepcopy(input_2d)
    input_t = {}
    for key in keys_3d:
        input_t[key] = input_3d[key][pick_ind, ...]
    input_t = deepcopy(input_t) # copied after ind subset
    
#     for key in keys_3d:
#         input_t[key][:, land_mask] = np.nan
    for key in keys_2d:
        input_z[key][land_mask] = np.nan
        
    # cropping detection
    Lx, Ly = land_mask.shape # domain size
    Nx = (Lx-size+1)//gap+1  # derive number of cropping by domain size
    Ny = (Ly-size+1)//gap+1
    start_flag = np.zeros([Nx, Ny]).astype(bool) # search the bottom left corner of croppings
    # check if croppings can match "ocean_f"
    for i in range(Nx):
        for j in range(Ny):
            N_ocean_grid = np.sum(land_mask[gap*i:gap*i+size, gap*j:gap*j+size])
            if N_ocean_grid>=0 and N_ocean_grid<=ocean_f*size*size:
                start_flag[i, j] = True
            else:
                start_flag[i, j] = False
    # crop inputs
    L = len(pick_ind)
    N_single = np.sum(start_flag) # largest possible samples in one day
    N_total = N_single*L          # largest possible samples in all days
    CROPPINGs = np.empty((N_total, size, size, len(keys_3d)+len(keys_2d)))
    count = 0 # exact number of samples
    for i in range(L):
        for indx in range(Nx):
            for indy in range(Ny):
                if start_flag[indx, indy]:
                    ind_xs = gap*indx; ind_xe = gap*indx+size
                    ind_ys = gap*indy; ind_ye = gap*indy+size
                    # adding random shifts <--- cannot larger than "gap"
                    if ind_xs > rnd_range and ind_xe < Lx-rnd_range:
                        d = np.random.randint(-1*rnd_range, rnd_range, dtype='int')
                        ind_xs += d; ind_xe += d
                    if ind_ys > rnd_range and ind_ye < Ly-rnd_range:
                        d = np.random.randint(-1*rnd_range, rnd_range, dtype='int')
                        ind_ys += d; ind_ye += d
                    # sparsity check 
                    test_frame = input_t['{}_REGRID'.format(var)][i, ind_xs:ind_xe, ind_ys:ind_ye]
                    N_zeros = np.sum(np.logical_or(np.isnan(test_frame), test_frame==0))
                    if N_zeros < sparse_f*size*size:
                        # selected input features
                        CROPPINGs[count, ..., 0] = input_t['{}_REGRID'.format(var)][i, ind_xs:ind_xe, ind_ys:ind_ye]
                        CROPPINGs[count, ..., 1] = input_z['etopo_regrid_clean'][ind_xs:ind_xe, ind_ys:ind_ye]
                        CROPPINGs[count, ..., 2] = input_z['etopo_regrid'][ind_xs:ind_xe, ind_ys:ind_ye]
                        count += 1
         
    print('\tNumber of croppings: {}'.format(count))
    return CROPPINGs[:count, ...]

def random_cropping_2d(input_2d, keys_2d, land_mask, size, gap, ocean_f=0.33, rnd_range=4):
    '''
    2d random cropping
    '''
    # copy dicts before mask out domains
    input_z = deepcopy(input_2d)
    for key in keys_2d:
        input_z[key][land_mask] = np.nan
        
    # cropping detection
    Lx, Ly = land_mask.shape # domain size
    Nx = (Lx-size+1)//gap+1  # derive number of cropping by domain size
    Ny = (Ly-size+1)//gap+1
    start_flag = np.zeros([Nx, Ny]).astype(bool) # search the bottom left corner of croppings
    # check if croppings can match "ocean_f"
    for i in range(Nx):
        for j in range(Ny):
            N_ocean_grid = np.sum(land_mask[gap*i:gap*i+size, gap*j:gap*j+size])
            if N_ocean_grid>=0 and N_ocean_grid<=ocean_f*size*size:
                start_flag[i, j] = True
            else:
                start_flag[i, j] = False
    # crop inputs
    N_total = np.sum(start_flag) # largest possible samples 
    CROPPINGs = np.empty((N_total, size, size, len(keys_2d))) # len(keys_2d)
    count = 0 # exact number of samples

    for indx in range(Nx):
        for indy in range(Ny):
            if start_flag[indx, indy]:
                ind_xs = gap*indx; ind_xe = gap*indx+size
                ind_ys = gap*indy; ind_ye = gap*indy+size
                # adding random shifts <--- cannot larger than "gap"
                if ind_xs > rnd_range and ind_xe < Lx-rnd_range:
                    d = np.random.randint(-1*rnd_range, rnd_range, dtype='int')
                    ind_xs += d; ind_xe += d
                if ind_ys > rnd_range and ind_ye < Ly-rnd_range:
                    d = np.random.randint(-1*rnd_range, rnd_range, dtype='int')
                    ind_ys += d; ind_ye += d
                # selected input features
                CROPPINGs[count, ..., 0] = input_z['etopo_4km'][ind_xs:ind_xe, ind_ys:ind_ye]
                CROPPINGs[count, ..., 1] = input_z['etopo_regrid'][ind_xs:ind_xe, ind_ys:ind_ye]
                count += 1
    print('\tNumber of croppings: {}'.format(count))
    return CROPPINGs[:count, ...]

def feature_norm(FEATUREs, method='norm_std', self_norm=False):
    '''
    feature normalization: norm_std, min_max, log_trans
    '''
    f_shape = FEATUREs.shape # L, (X, Y), C
    
    # select normalization functions from data_utils
    f = eval(method)
    # self norm: normalizing each channel independently (otherwise norm target by input)
    if self_norm:
        for i in range(f_shape[0]):
            for j in range(0, f_shape[-1]):
                FEATUREs[i, ..., j] = fillzero(f(FEATUREs[i, ..., j]))
    else: 
        for i in range(f_shape[0]):
            if method == 'norm_std':
                grid_mean1 = np.nanmean(FEATUREs[i, ..., 1])
                grid_std1 = np.nanstd(FEATUREs[i, ..., 1])
                FEATUREs[i, ..., 0] = fillzero((FEATUREs[i, ..., 0]-grid_mean1)/(grid_std1))
            elif method == 'min_max':
                grid_min1 = np.nanmin(FEATUREs[i, ..., 1])
                grid_max1 = np.nanmax(FEATUREs[i, ..., 1])
                FEATUREs[i, ..., 0] = fillzero((FEATUREs[i, ..., 0]-grid_min1)/(grid_max1-grid_min1))
            else:
                raise ValueError('Wrong keywords')

            for j in range(1, f_shape[-1]):
                FEATUREs[i, ..., j] = fillzero(f(FEATUREs[i, ..., j]))
    return FEATUREs

def batch_gen(FEATUREs, batch_size, BATCH_dir, perfix, ind0=0):
    '''
    Spliting and generating batches as naive numpy files
        - perfix: filename
        - ind0: the start of batch index
        e.g., 'perfix0.npy' given ind0=0
    '''
    L = len(FEATUREs)
    N_batch = L//batch_size # losing some samples
    
    print('\tNumber of batches: {}'.format(N_batch))
    # shuffle
    ind = shuffle_ind(L)
    FEATUREs = FEATUREs[ind, ...]
    print("\tShuffling and save batches ('X', 'Y')")
    # loop
    for i in range(N_batch):
        save_d = {'batch':FEATUREs[batch_size*i:batch_size*(i+1), ...]}
        temp_name = BATCH_dir+perfix+str(ind0+i)+'.npy'
        print(temp_name) # print out saved filenames
        np.save(temp_name, save_d)
    return N_batch

