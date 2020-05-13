import sys
import h5py
import numpy as np
from glob import glob
from datetime import datetime, timedelta
from scipy.stats import gamma

sys.path.insert(0, '/glade/u/home/ksha/ML_repo/utils/')
import PRISM_utils as pu
import data_utils as du
import keras_utils as ku
import graph_utils as gu
from namelist_PRISM import * 

hist_dir = '/glade/work/ksha/data/Keras/PRISM_publish/'

seasons = ['DJF', 'MAM', 'JJA', 'SON']

base = datetime(2018, 1, 1, 0)
date_list = [base + timedelta(days=x) for x in range(364)]

IND = {}; L = len(date_list)
base_ind0 = np.zeros(L).astype(bool)
base_ind1 = np.zeros(L).astype(bool)
base_ind2 = np.zeros(L).astype(bool)
base_ind3 = np.zeros(L).astype(bool)

for i, dt in enumerate(date_list):
    m = dt.month
    if m==1 or m==2 or m==12:
        base_ind0[i] = True
    elif m==3 or m==4 or m==5:
        base_ind1[i] = True
    elif m==6 or m==7 or m==8:
        base_ind2[i] = True
    else:
        base_ind3[i] = True

IND[seasons[0]] = base_ind0
IND[seasons[1]] = base_ind1
IND[seasons[2]] = base_ind2
IND[seasons[3]] = base_ind3
IND['annual'] = np.ones(L).astype(bool) # <----

# generate rain/np-rain flags by threholds
## rain prob by reliability diagram
hdf_io = h5py.File(BACKUP_dir+'PRISM_pct_2015_2018.hdf', 'r')
PRISM_TEST = hdf_io['PRISM_PCT'][365+366+365:-2, ...]
hdf_io.close()
PRISM_TEST = np.exp(PRISM_TEST)-1 # <---- convert
# land_mask
land_mask = np.isnan(PRISM_TEST[0, ...])
pick_flag = ~land_mask

SD = {}
models  = ['PCT_REGRID', 'UNET_A', 'UNET_B', 'XNET_A', 'XNET_B']
hdf_io = h5py.File(hist_dir+'PRISM_PRED_NCEP_PCT_2016_2018.hdf', 'r')
for model in models:
    temp_val = np.exp(hdf_io[model][...])-1# <---- convert
    temp_val[:, land_mask] = np.nan
    SD[model] = temp_val[-364:]
hdf_io.close()
SD['TRUE'] = PRISM_TEST
print('size: {} v.s. {}'.format(SD[model].shape, SD['TRUE'].shape))

# gamma fit
thres = 60.0
pct_bins = np.linspace(thres, 180, 100)

HIST = {}
seasons = ['DJF', 'MAM', 'JJA', 'SON', 'annual']
models  = ['TRUE', 'PCT_REGRID', 'UNET_A', 'XNET_A']

pick_flag = ~land_mask


for i, sea in enumerate(seasons):
    print(sea)
    for j, model in enumerate(models):
        print('\t{}'.format(model))
        temp_true = SD[model][IND[sea], ...][:, pick_flag].ravel()
        if model == 'PCT_REGRID':
            temp_true = 0.9*temp_true
        temp_true = temp_true[temp_true>thres]
        temp_hist, _ = np.histogram(temp_true, pct_bins)
        HIST['{}_{}'.format(model, sea)] = temp_hist
        HIST['{}_{}_param'.format(model, sea)] = [np.std(temp_true), np.percentile(temp_true, 99)]

        alpha, loc, beta = gamma.fit(temp_true)
        data = gamma.rvs(alpha, loc=loc, scale=beta, size=len(temp_true))
        temp_hist, _ = np.histogram(data, pct_bins, density=True)
        HIST['{}_{}_gamma'.format(model, sea)] = temp_hist
        HIST['{}_{}_gamma_param'.format(model, sea)] = [alpha, loc, beta]
        
np.save(hist_dir+'PCT_FNL_gamma_fit.npy', HIST)


