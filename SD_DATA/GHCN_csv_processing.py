# general tools
import sys
from glob import glob
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np
import pandas as pd
# stats tools

from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/')
from namelist import *
import data_utils as du


with pd.HDFStore(OBS_dir+'metadata.hdf', 'r') as hdf_temp:
    metadata_USBC = hdf_temp['metadata_USBC']

stn_domain = metadata_USBC['stn code'].values
for year in ['2020']:
    print('Processing year {}'.format(year))
    obs2020 = pd.read_csv(OBS_CSV_dir+'{}.csv'.format(year), header=None)
    obs2020.columns = ['stn code', 'date', 'variable', 'value', 'measurement flag', 'quality flag', 'source flag', 'obstime']
    obs2020 = obs2020[obs2020['variable'].str.contains('TAVG')]
    qc_flag = np.logical_not(obs2020['quality flag'].map(type).eq(str).values)
    obs2020_qc = obs2020.iloc[qc_flag]
    
    obs2020_qc = obs2020_qc.drop('variable', 1)
    obs2020_qc = obs2020_qc.drop('measurement flag', 1)
    obs2020_qc = obs2020_qc.drop('quality flag', 1)
    obs2020_qc = obs2020_qc.drop('source flag', 1)
    obs2020_qc = obs2020_qc.drop('obstime', 1)
    
    print('Selecting US/BC stations')   
    stns = obs2020_qc['stn code'].values
    domain_flag = np.zeros(len(stns)).astype(bool)

    for i, stn in enumerate(stns):
        if stn in stn_domain:
            domain_flag[i] = True

    obs2020_BCUS = obs2020_qc.iloc[domain_flag]
    dict_obs2020_BCUS = dict(tuple(obs2020_BCUS.groupby('stn code')))
    
    hdf_keys = list(dict_obs2020_BCUS.keys())
    with pd.HDFStore(OBS_dir+'obs{}.hdf'.format(year), 'w') as hdf_temp:
        for hdf_key in hdf_keys:
            print('Save {}'.format(hdf_key))
            temp_pd = dict_obs2020_BCUS[hdf_key]
            # datetime conversion
            temp_pd['date'] = pd.to_datetime(temp_pd['date'], format='%Y%m%d')
            temp_pd['date'] = temp_pd['date'].dt.date.values
            #
            hdf_temp[hdf_key] = dict_obs2020_BCUS[hdf_key]
