
import sys
import h5py
import numpy as np
#import netCDF4 as nc
from glob import glob
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/ML_repo/utils/')
import PRISM_utils as pu
import data_utils as du
import keras_utils as ku
import graph_utils as gu
from namelist_PRISM import * 

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib import ticker
from matplotlib.collections import PatchCollection
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib import ticker
from mpl_toolkits.axes_grid.inset_locator import inset_axes, zoomed_inset_axes, mark_inset, InsetPosition

from joblib import dump, load
from matplotlib.colors import LinearSegmentedColormap

fig_dir = '/glade/u/home/ksha/figures/'
data_dir = '/glade/work/ksha/data/Keras/PRISM_publish/'
hist_dir = '/glade/work/ksha/data/Keras/PRISM_publish/'

from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix

def ETS(TRUE, PRED):
    TN, FP, FN, TP = confusion_matrix(TRUE, PRED).ravel()
    TP_rnd = (TP+FN)*(TP+FP)/(TN+FP+FN+TP)
    return (TP-TP_rnd)/(TP+FN+FP-TP_rnd)

def freq_bias(TRUE, PRED):
    TN, FP, FN, TP = confusion_matrix(TRUE, PRED).ravel()
    return (TP+FP)/(TP+FN)

seasons = ['DJF', 'MAM', 'JJA', 'SON']

base = datetime(2018, 1, 1, 0)
date_list = [base + timedelta(days=x) for x in range(366)]

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

OUT = {}
models  = ['UNET_A', 'UNET_B', 'XNET_A', 'XNET_B']
hdf_io = h5py.File('/glade/work/ksha/data/Keras/PRISM_publish/PRISM_PCT_flags_2018.hdf', 'r')
for model in models:
    OUT[model] = hdf_io[model][...]
hdf_io.close()

hdf_io = h5py.File(BACKUP_dir+'PRISM_pct_2015_2018.hdf', 'r')
PRISM_TEST = hdf_io['PRISM_PCT'][-366:, ...]
hdf_io.close()

TRUE_flag = PRISM_TEST > 0

hdf_io = h5py.File(BACKUP_dir+'PRISM_pct_2015_2018.hdf', 'r')
land_mask = np.isnan(hdf_io['PRISM_PCT'][0, ...])
etopo = hdf_io['etopo'][...]
lon = hdf_io['lon'][...]
lat = hdf_io['lat'][...]
hdf_io.close()

# thres_val[season][model]
seasons = ['DJF', 'MAM', 'JJA', 'SON']
models  = ['UNET_A', 'UNET_B', 'XNET_A', 'XNET_B']
#thres_val = [[0.71, 0.72, 0.67, 0.5], [0.78, 0.71, 0.74, 0.5], [0.73, 0.68, 0.73, 0.72], [0.69, 0.71, 0.65, 0.73]] # 0.76
thres_test = np.arange(0.5, 0.8, 0.02)

CMap = plt.cm.nipy_spectral_r
JET = [CMap(50), CMap(100), CMap(150), CMap(200)]

for nn, test in enumerate(thres_test):
    print(nn)
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    FLAG_RESULT = {}
    for i, model in enumerate(models):
        temp = np.zeros(OUT[model].shape).astype(bool)
        for j, sea in enumerate(seasons):    
            temp[IND[sea], ...] = OUT[model][IND[sea], ...] >= test#thres_val[j][i]
        FLAG_RESULT[model] = temp

    result = {}
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    models  = ['UNET_A', 'UNET_B', 'XNET_A', 'XNET_B']
    pick_flag = ~land_mask
    for model in models:
        true_temp = TRUE_flag[:, pick_flag].ravel()
        pred_temp = FLAG_RESULT[model][:, pick_flag].ravel()
        result['ETC_{}_annual'.format(model)] = ETS(true_temp, pred_temp)
        result['freq_bias_{}_annual'.format(model)] = freq_bias(true_temp, pred_temp)
        result['jaccard_{}_annual'.format(model)] = jaccard_score(true_temp, pred_temp, pos_label=1, average='binary')
        for sea in seasons:
            true_temp = TRUE_flag[:, pick_flag][IND[sea], :].ravel()
            pred_temp = FLAG_RESULT[model][:, pick_flag][IND[sea], :].ravel()
            result['ETC_{}_{}'.format(model, sea)] = ETS(true_temp, pred_temp)
            result['freq_bias_{}_{}'.format(model, sea)] = freq_bias(true_temp, pred_temp)
            result['jaccard_{}_{}'.format(model, sea)] = jaccard_score(true_temp, pred_temp, pos_label=1, average='binary')

    models  = ['UNET_A', 'UNET_B', 'XNET_A', 'XNET_B']
    metrics = ['ETC', 'freq_bias', 'jaccard'] #, 
    seasons = ['DJF', 'MAM', 'JJA', 'SON', 'annual']

    fig, AX = plt.subplots(3, 5, figsize=(13, 5))
    plt.subplots_adjust(0, 0, 1, 1, hspace=0.75, wspace=0.25)

    edge_loc = 0.7
    bar_gap = 0.125

    base = [0.69, 0.75, 0.75]
    for i, metric in enumerate(metrics):
        for j, sea in enumerate(seasons):
            ax = AX[i][j]
            #Y_MEAN = np.mean(M[ind[j], :, i])
            #d = ylim_ratio[i, j]*Y_MEAN
            ax.set_ylim([base[i], base[i]+0.25])
            ax.set_xlim([-0.5*bar_gap, 3.5*bar_gap])
            #
            ax = gu.ax_decorate(ax, False, False)
            ax.grid(False)
            ax.spines["bottom"].set_visible(True)
            ax.spines["left"].set_visible(False)
            for k, model in enumerate(models):
                temp_val = np.array([result['{}_{}_{}'.format(metric, model, sea)]])
                marker_p, stem_p, base_p = ax.stem(np.array([0+bar_gap*k]), temp_val)
                plt.setp(marker_p, marker='s', ms=15, mew=2.5, mec='k', mfc=JET[k], zorder=4)
                plt.setp(stem_p, linewidth=2.5, color='k')
                ax.text(np.array([0+bar_gap*k]), temp_val+0.05, str(np.round(temp_val[0], 3)), 
                        va='center', ha='center', fontsize=12)
    AX[0][1].set_title(str(test), fontsize=20)
    fig.savefig(fig_dir+'PCT_test_{}.png'.format(nn), dpi=250, orientation='portrait', \
                papertype='a4', format='png', bbox_inches='tight', pad_inches=0.1)