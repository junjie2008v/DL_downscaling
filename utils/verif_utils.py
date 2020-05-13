# general tools
import sys
import glob

# data tools
import random
import numpy as np

# science tools
import metpy.calc
from metpy.units import units
from sklearn.linear_model import LinearRegression, Lasso

# deep learning tools
import tensorflow as tf
from tensorflow import keras
#import tensorflow.keras.backend as K

sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
from data_utils import *

def norm_tuple(X, method='norm_std'):
    '''
    normalizing all elements of a tuple
    '''
    f = eval(method)
    L = len(X)
    OUT = ()
    for i in range(L):
        OUT += (f(X[i]),)
    return OUT

def mask_tuple(X, mask):
    '''
    mask tuple element (not copied)
    '''
    L = len(X)
    OUT = ()
    for i in range(L):
        temp = X[i]
        temp[mask] = np.nan
        OUT += (temp,)
    return OUT

# ---------- CNN utils ---------- #

def pred_domain(X, land_mask, model, param, method, ind=0):
    '''
    Full domain inference with overlapped tiles
    '''
    L = len(X)
    gap = param['gap']
    size = param['size']
    edge = param['edge']
    size_center = size-2*edge;
    
    grid_shape = land_mask.shape
    
    # tile number estimation
    Ny = int(np.ceil((grid_shape[0])/size_center))
    Nx = int(np.ceil((grid_shape[1])/size_center))+1
    grid_shape_pad = (Ny*(size_center)+2*edge, Nx*(size_center)+2*edge)
    Px, Py = (grid_shape_pad[1]-grid_shape[1]-edge)//gap, (grid_shape_pad[0]-grid_shape[0]-edge)//gap
    zero_pad = np.zeros(grid_shape_pad+(L,))*np.nan
    
    # land mask operation
    X = mask_tuple(X, land_mask)

    # pred tiles (in a loop for allowing multi-tile segmentation)
    count = 0
    for i in range(Py):
        for j in range(Px):
            if count == ind:
                for k in range(L):
                    zero_pad[edge+gap*i:edge+gap*i+grid_shape[0], edge+gap*j:edge+gap*j+grid_shape[1], k] = X[k]
                return pred_tile(zero_pad, model, param, Nx, Ny, method)[edge+gap*i:edge+gap*i+grid_shape[0], edge+gap*j:edge+gap*j+grid_shape[1]]
            else:
                count += 1
    print('ind not found')

def pred_tile(data, model, param, Nx, Ny, method, zero_thres=64):
    '''
    zero_thres: minimum number of non-zero values in a tile
    '''
    f = eval(method)
    f_inv = eval('inv_'+method)
    # params
    size = param['size']
    edge = param['edge']
    size_center = size-2*edge
    # allocations
    grid_shape = data.shape
    out = np.zeros([grid_shape[0], grid_shape[1]])
    temp_input = np.zeros([size, size, grid_shape[2]])
    # 
    for i in range(Ny):
        for j in range(Nx):
            # indices
            indy_start = i*size_center
            indx_start = j*size_center
            indyc_start = indy_start+edge
            indxc_start = indx_start+edge
            
            # a single tile
            temp_data = data[indy_start:indy_start+size, indx_start:indx_start+size, :]
            # test zero_thres
            flag_effective = np.logical_and(~np.isnan(temp_data[..., 0]), np.abs(temp_data[..., 0])>0)
            N_effective = np.sum(flag_effective)
            if N_effective < 64 :
                # all nans
                out[indyc_start:indyc_start+size_center, indxc_start:indxc_start+size_center] = np.zeros([size_center, size_center])
            # pass to the keras model
            else:
                if method == 'norm_std':
                    norm_param1 = np.nanmean(temp_data[..., 0])
                    norm_param2 = np.nanstd(temp_data[..., 0])
                elif method == 'min_max':
                    norm_param1 = np.nanmin(temp_data[..., 0])
                    norm_param2 = np.nanmax(temp_data[..., 0])
                
                for k in range(grid_shape[2]):
                    temp_feature = temp_data[..., k]
                    temp_input[..., k] = fillzero(f(temp_feature))
                
                out_temp = model.predict([temp_input[None, ...]])
                if isinstance(out_temp, list):
                    out_temp = np.squeeze(out_temp[0])
                else:
                    out_temp = np.squeeze(out_temp)
                    
                out[indyc_start:indyc_start+size_center, indxc_start:indxc_start+size_center] = f_inv(out_temp, norm_param1, norm_param2)[edge:size-edge, edge:size-edge]
    return out

# ---------- baseline utils ---------- #

def linregress_train(X_3d, X_2d, Y, land_mask):
    '''
    Estimating grid-point-wise linear coefficients
        skipping land_mask==True grid points
        *All-zero X_3d input grid point triggers warning
        *All-zero X_2d input grid point may trigger warning as well
    '''
    ocean_mask = np.logical_not(land_mask)
    grid_shape = land_mask.shape
    grid_shape3d = X_3d[0].shape
    
    L3d = len(X_3d)
    L2d = len(X_2d)
    L_all = L3d + L2d
    
    I = np.zeros(grid_shape) # intercept
    C = np.zeros(grid_shape+(L_all,)) # coef

    # duplicate 2d fields to 3d for training
    X_input = ()
    for i in range(L2d):
        temp_3d = np.repeat(X_2d[i][None, ...], grid_shape3d[0], axis=0)
        X_input += (temp_3d,)
    # combining 2d and 3d fields     
    X_input = X_3d + X_input

    # grid-point-wise operation
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if ocean_mask[i, j]:
                temp_tuple = () # python tuple is **ordered**, use tuple for np.concat 
                for k in range(L_all):
                    temp_tuple += (X_input[k][:, i, j][:, None],)                
                X_series = np.concatenate(temp_tuple, axis=1) # X[sample, feature]
                # sklearn Ridge estimator
                gamma = Lasso()
                # fit and collect params
                gamma.fit(X_series, Y[:, i, j][:, None])
                I[i, j] = gamma.intercept_
                C[i, j, :] = gamma.coef_
    return I, C

def linregress_pred(I, C, X_3d, X_2d, land_mask):
    '''
    Predictions by grid-point-wise linear coefficients
    '''
    ocean_mask = np.logical_not(land_mask)
    grid_shape = land_mask.shape
    grid_shape3d = X_3d[0].shape
    
    OUT = np.zeros(grid_shape3d)*np.nan
    
    L3d = len(X_3d)
    L2d = len(X_2d)
    L_all = L3d + L2d
    
    # duplicate 2d fields to 3d for prediction
    X_input = ()
    for i in range(L2d):
        temp_3d = np.repeat(X_2d[i][None, ...], grid_shape3d[0], axis=0)
        X_input += (temp_3d,)
    # combining 2d and 3d fields     
    X_input = X_3d + X_input
    
    # grid-point-wise operation
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if ocean_mask[i, j]:
                temp_tuple = () # python tuple is **ordered**, use tuple for np.concat 
                for k in range(L_all):
                    temp_tuple += (X_input[k][:, i, j][:, None],)                
                X_series = np.concatenate(temp_tuple, axis=1) # X[sample, feature]
                OUT[:, i, j] = X_series.dot(C[i, j, :])+I[i, j]
    return OUT

def baseline_estimator(X_3d, X_2d, Y, X_3d_pred, X_2d_pred, land_mask):
    '''
    Temperature downscaling baseline
    '''
#     # Found normalization is not necessary
#     tstd = np.nanstd(X_3d[0], axis=0)
#     tmean = np.nanmean(X_3d[0], axis=0)
#     X_3d = norm_tuple(X_3d, method=method)
#     X_2d = norm_tuple(X_2d, method=method)
#     Y = norm_std(Y)
#     X_3d_pred = norm_tuple(X_3d_pred, method=method) 
#     X_2d_pred = norm_tuple(X_2d_pred, method=method)
    
    I, C = linregress_train(X_3d, X_2d, Y, land_mask)
    Y_pred = linregress_pred(I, C, X_3d_pred, X_2d_pred, land_mask)
    
    return I, C, Y_pred

# ----- evaluation functions ----- #

def hist2dxy(hist_x, hist_y, MIN, MAX, Xnum, Ynum, density=False):
    '''
    2d histograms calculation
    X, Y, H = hist2dxy()
    with output supports plt.pcolor(X, Y, H)
    
    hist_x, hist_y: x-axis and y-axis variables in a 2d-histogram
    MIN, MAX: [min, max] range of the hist (for both x, y)
    Xnum, Ynum: number of bins
    density: False = output bin counts; True = output prob.
    '''
    if isinstance(MIN, list):
        Xlim = [MIN[0], MAX[0]]
        Ylim = [MIN[1], MAX[1]]
    else:
        Xlim = [MIN, MAX]
        Ylim = [MIN, MAX]

    Xbins = np.linspace(Xlim[0], Xlim[1], Xnum, dtype=np.float)
    Ybins = np.linspace(Ylim[0], Ylim[1], Ynum, dtype=np.float)
    H, y_edges, x_edges = np.histogram2d(hist_y, hist_x, bins=(Ybins, Xbins), density=density)
    X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])
    return X, Y, H

def hybrid_to_m(slp=100000):
    '''
    For the first 20 hybrid levs
    slp unit = Pa
    height unit = m
    '''
    # hybrid lev 1 to 21
    A = np.array([0, 0, 0, 0, 0, 0, 0, 0, 133.051011276943, 364.904148871589, 634.602716447362,
                  959.797167291774, 1347.68004165515, 1790.9073959511, 2294.8416899485, 2847.48477771176,
                  3468.87148811864, 4162.95646296916, 4891.88083250491, 5671.82423980408, 6476.71299638532])

    B = np.array([1.0, 0.997, 0.994, 0.989, 0.982, 0.972, 0.96, 0.946, 0.926669489887231, 0.904350958511284,
                  0.879653972835526, 0.851402028327082, 0.819523199583449, 0.785090926040489, 0.748051583100515,
                  0.709525152222882, 0.668311285118814, 0.624370435370308, 0.580081191674951, 0.534281757601959,
                  0.488232870036147])
    half_lev = A + B*slp
    full_lev = np.zeros([20])
    for i in range(20):
        full_lev[i] = np.exp(1/(half_lev[i]-half_lev[i+1])*
                             (half_lev[i]*np.log(half_lev[i])-half_lev[i+1]*np.log(half_lev[i+1]))-1)
    height = metpy.calc.pressure_to_height_std(full_lev*units.Pa).__array__()
    height = height*1000
    return full_lev, height
