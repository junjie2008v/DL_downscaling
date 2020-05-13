import sys
import h5py
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

sys.path.insert(0, '/glade/u/home/ksha/ML_repo/utils/')
import PRISM_utils as pu
import data_utils as du
import keras_utils as ku
import graph_utils as gu
from namelist_PRISM import * 

# rain prob by reliability diagram\
hdf_io = h5py.File(BACKUP_dir+'PRISM_pct_2015_2018.hdf', 'r')
PRISM_TEST = hdf_io['PRISM_PCT'][-366:, ...]
REGRID_TEST = hdf_io['PCT_REGRID'][-366:, ...]
PRISM_TRAIN = hdf_io['PRISM_PCT'][:-366, ...]
REGRID_TRAIN = hdf_io['PCT_REGRID'][:-366, ...]
hdf_io.close()

land_mask = np.isnan(PRISM_TRAIN[0, ...])

LR_CDF = np.zeros((399,)+land_mask.shape)*np.nan
HR_CDF = np.zeros((399,)+land_mask.shape)*np.nan

pct_bins = np.linspace(0, 200, 400)

for i in range(600):
    print(i)
    for j in range(600):
        if ~land_mask[i, j]:
            LR_series = REGRID_TRAIN[:, i, j]
            HR_series = PRISM_TRAIN[:, i, j]
            if np.sum(~np.isnan(LR_series)) > 365:

                LR_CDF[:, i, j] = plt.hist(LR_series, bins=pct_bins, cumulative=True, density=True)[0];
                HR_CDF[:, i, j] = plt.hist(HR_series, bins=pct_bins, cumulative=True, density=True)[0];

REGRID_QM = np.zeros(REGRID_TEST.shape)*np.nan

for t in range(366):
    print('t: {}'.format(t))
    for i in range(600):
        for j in range(600):
            LR_test = REGRID_TEST[t, i, j]
            if ~np.isnan(HR_CDF[0, i, j]) and ~np.isnan(LR_test):
                ind = np.searchsorted(pct_bins, LR_test, 'left')
                REGRID_QM[t, i, j] = HR_CDF[ind, i, j]

dict_s = {'QM':REGRID_QM, 'LR_CDF':LR_CDF, 'HR_CDF':HR_CDF}
np.save('/glade/work/ksha/data/Keras/PRISM_publish/PRISM_QM.npy', dict_s)
