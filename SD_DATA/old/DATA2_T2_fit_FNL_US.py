import sys
import h5py
import numpy as np
from glob import glob

sys.path.insert(0, '/glade/u/home/ksha/ML_repo/utils/')
import data_utils as du
from namelist_PRISM import *
import scipy.optimize

def sinfunc(t, A, w, p, b, c):
    return A * np.sin(w*t + p) + b*t + c

def fit_sin(t, y):
    w0 = 1.0 # <--- one year
    amp0 = np.std(y) * 2.**0.5
    k0 = 0.1
    offset0 = np.mean(y)
    
    guess = np.array([amp0, w0, 0, k0, offset0])
    popt, _ = scipy.optimize.curve_fit(sinfunc, t, y, p0=guess, maxfev=100000000)
    return popt

VARS = ['TMAX', 'TMIN']
time_sec = np.concatenate((np.linspace(0, 2*np.pi, 366), np.linspace(0, 2*np.pi, 365), np.linspace(0, 2*np.pi, 365)), axis=0)
time_sec = time_sec[:-1]
leap = [True, False, False]
for VAR in VARS:
    print(VAR)
    hdf_io = h5py.File(BACKUP_dir+'NCEP_FNL_2016_2018_4km.hdf', 'r')
    REGRID_VAR = hdf_io['{}_C'.format(VAR)][:, ...]
    hdf_io.close()
    _, X1, X2 = REGRID_VAR.shape

    print('Moving window mean')
    gap = 5
    year_ind = np.arange(0, 365+gap, gap)
    feb_gap = ((31+28)//gap)+1
    leap_year_ind = np.hstack([np.arange(0, feb_gap*gap, gap), np.arange(feb_gap*gap+1, 366+gap, gap)])
    start_id = [0, 366, 366+365]

    L = len(year_ind)-1
    MEAN = np.zeros([L, X1, X2])*np.nan
    leap_year_flag = [False, True, False]
    for i in range(X1):
        for j in range(X2):
            for n in range(L):
                temp_series = np.array([])
                for s, inds in enumerate(start_id):
                    if leap_year_flag[s]:
                        temp_series = np.concatenate((temp_series, REGRID_VAR[s+leap_year_ind[n]:s+leap_year_ind[n+1], i, j]), axis=0)
                    else:
                        temp_series = np.concatenate((temp_series, REGRID_VAR[s+year_ind[n]:s+year_ind[n+1], i, j]), axis=0)
                MEAN[n, i, j] = np.mean(temp_series)
    MEAN_BASE = np.zeros([365, X1, X2])*np.nan    
    MEAN_BASE_leap = np.zeros([366, X1, X2])*np.nan

    for i in range(L):
        MEAN_BASE[year_ind[i]:year_ind[i+1], ...] = np.repeat(MEAN[i, ...][None, ...], gap, axis=0)
        MEAN_BASE_leap[leap_year_ind[i]:leap_year_ind[i+1], ...] = np.repeat(MEAN[i, ...][None, ...], 
                                                                             leap_year_ind[i+1]-leap_year_ind[i], axis=0)
    FIT_MEAN = np.zeros([365, X1, X2])*np.nan
    FIT_MEAN_leap = np.zeros([366, X1, X2])*np.nan
    time_base = np.linspace(0, 2*np.pi, 365)
    time_base_leap = np.linspace(0, 2*np.pi, 366)
    print('Fit mean')
    for i in range(X1):
        for j in range(X2):
            #if .. land_mask 
            fit_base = np.zeros([365])
            fit_base_leap = np.zeros([366])
            count = 0
            A, w, p, b, c = fit_sin(time_base, MEAN_BASE[:, i, j])
            FIT_MEAN[:, i, j] = sinfunc(time_base, A, w, p, b, c)
            A, w, p, b, c = fit_sin(time_base_leap, MEAN_BASE_leap[:, i, j])
            FIT_MEAN_leap[:, i, j] = sinfunc(time_base_leap, A, w, p, b, c)

    FIT_T = np.concatenate((FIT_MEAN_leap, FIT_MEAN, FIT_MEAN[:-1, ...]), axis=0)
    REGRID_VAR = REGRID_VAR-FIT_T
    print('Moving window std')
    STD = np.zeros([L, X1, X2])*np.nan
    for i in range(X1):
        for j in range(X2):
            for n in range(L):
                temp_series = np.array([])
                for s, inds in enumerate(start_id):
                    if leap_year_flag[s]:
                        temp_series = np.concatenate((temp_series, REGRID_VAR[s+leap_year_ind[n]:s+leap_year_ind[n+1], i, j]), axis=0)
                    else:
                        temp_series = np.concatenate((temp_series, REGRID_VAR[s+year_ind[n]:s+year_ind[n+1], i, j]), axis=0)
                STD[n, i, j] = np.std(temp_series)
                    
    STD_BASE  = np.zeros([365, X1, X2])*np.nan
    STD_BASE_leap  = np.zeros([366, X1, X2])*np.nan
    for i in range(L):
        STD_BASE[year_ind[i]:year_ind[i+1], ...] = np.repeat(STD[i, ...][None, ...], gap, axis=0)
        STD_BASE_leap[leap_year_ind[i]:leap_year_ind[i+1], ...] = np.repeat(STD[i, ...][None, ...], leap_year_ind[i+1]-leap_year_ind[i], axis=0)
    print('Fit std')
    FIT_STD = np.zeros([365, X1, X2])*np.nan
    FIT_STD_leap = np.zeros([366, X1, X2])*np.nan
    for i in range(X1):
        for j in range(X2):
            #if .. land_mask 
            fit_base = np.zeros([365])
            fit_base_leap = np.zeros([366])
            count = 0
            A, w, p, b, c = fit_sin(time_base, STD_BASE[:, i, j])
            FIT_STD[:, i, j] = sinfunc(time_base, A, w, p, b, c)
            A, w, p, b, c = fit_sin(time_base_leap, STD_BASE_leap[:, i, j])
            FIT_STD_leap[:, i, j] = sinfunc(time_base_leap, A, w, p, b, c)

    save_dict = {'FIT_MEAN':FIT_MEAN, 'FIT_MEAN_leap':FIT_MEAN_leap,
                 'FIT_STD' :FIT_STD , 'FIT_STD_leap' :FIT_STD_leap}

    np.save(BACKUP_dir+'{}_NCEP_STD.npy'.format(VAR), save_dict)
    print('Save to: {}'.format(BACKUP_dir+'{}_NCEP_STD.npy'.format(VAR)))

