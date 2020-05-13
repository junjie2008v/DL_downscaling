import sys
#import h5py
import numpy as np
#import pandas as pd
from glob import glob
import tensorflow as tf
from tensorflow import keras
#from datetime import datetime
#import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

#from random import shuffle
import random

sys.path.insert(0, '/glade/u/home/ksha/ML_repo/utils/')
import PRISM_utils as pu
import data_utils as du
import keras_utils as ku
from namelist_PRISM import *

class grid_grid_gen(keras.utils.Sequence):
    def __init__(self, file_names, batch_size, c_size, m_size, labels, shuffle_ind, flag=[True, True]):
        self.batch_size = batch_size # typically 4, 8, 16 etc.
        self.c_size = c_size
        self.m_size = m_size
        self.labels = labels
        self.flag = flag
        self.file_names = file_names
        self.shuffle_ind = shuffle_ind
        self.file_len = len(self.file_names)
        self.inds = np.arange(self.file_len).astype(int)
    def __len__(self):
        return int(np.floor(self.file_len/self.batch_size))
    
    def __getitem__(self, index):
        random.shuffle(self.file_names)
        temp_file_names = self.file_names[index*self.batch_size:(index+1)*self.batch_size]
        return self.__readfile(temp_file_names, self.c_size, self.m_size, self.labels, self.shuffle_ind, self.flag)
    
    def __readfile(self, names, c_size, m_size, labels, shuffle_ind, flag):
        N = len(names)
        data_temp = np.load(names[0])
        GRID_IN = data_temp[()][labels[0]][...]
        Y_LABEL = data_temp[()][labels[1]][..., None]
        Y2 = np.copy(-1*GRID_IN[..., 1][..., None])
        rnd_ind = du.shuffle_ind(200)
        GRID_IN[..., shuffle_ind] = GRID_IN[rnd_ind, ..., shuffle_ind]
        return [GRID_IN], [Y_LABEL, Y2]
    
class grid_grid_gen3(keras.utils.Sequence):
    def __init__(self, file_names, batch_size, c_size, m_size, labels, shuffle_ind, flag=[True, True]):
        self.batch_size = batch_size # typically 4, 8, 16 etc.
        self.c_size = c_size
        self.m_size = m_size
        self.labels = labels
        self.flag = flag
        self.file_names = file_names
        self.shuffle_ind = shuffle_ind
        self.file_len = len(self.file_names)
        self.inds = np.arange(self.file_len).astype(int)
    def __len__(self):
        return int(np.floor(self.file_len/self.batch_size))
    
    def __getitem__(self, index):
        random.shuffle(self.file_names)
        temp_file_names = self.file_names[index*self.batch_size:(index+1)*self.batch_size]
        return self.__readfile(temp_file_names, self.c_size, self.m_size, self.labels, self.shuffle_ind, self.flag)
    
    def __readfile(self, names, c_size, m_size, labels, shuffle_ind, flag):
        N = len(names)
        data_temp = np.load(names[0])
        GRID_IN = data_temp[()][labels[0]][..., flag]
        Y_LABEL = data_temp[()][labels[1]][..., None]
        
        rnd_ind = du.shuffle_ind(200)
        GRID_IN[..., shuffle_ind] = GRID_IN[rnd_ind, ..., shuffle_ind]
        
        return [GRID_IN], Y_LABEL

VARS = ['TMAX', 'TMIN']
L = 25; c_size = 2; labels = ['X', 'Y']; flag_c = [True, True, True]
save_dir = '/glade/work/ksha/data/Keras/PRISM_publish/'
for VAR in VARS:
    print(VAR)
    model_unet_a = keras.models.load_model(save_dir+'UNET_{}_A_std'.format(VAR)+'.hdf')
    model_unet_c = keras.models.load_model(save_dir+'UNET_{}_B_tune'.format(VAR)+'.hdf')

    train_files = glob(BATCH_dir+'{}*BATCH*128*TORI[0-9]*.npy'.format(VAR))
    L_train, batch_size, m_size = len(train_files), 1, 200
    steps = L_train//batch_size

    train_gen = ku.grid_grid_gen(train_files, batch_size, c_size, m_size, labels, flag=flag_c)
    train_gen_LRT = grid_grid_gen(train_files, batch_size, c_size, m_size, labels, 0, flag=flag_c)
    train_gen_HRZ = grid_grid_gen(train_files, batch_size, c_size, m_size, labels, 1, flag=flag_c)
    train_gen_LRZ = grid_grid_gen(train_files, batch_size, c_size, m_size, labels, 2, flag=flag_c)

    train_gen3 = ku.grid_grid_gen3(train_files, batch_size, c_size, m_size, labels, flag=flag_c)
    train_gen3_LRT = grid_grid_gen3(train_files, batch_size, c_size, m_size, labels, 0, flag=flag_c)
    train_gen3_HRZ = grid_grid_gen3(train_files, batch_size, c_size, m_size, labels, 1, flag=flag_c)
    train_gen3_LRZ = grid_grid_gen3(train_files, batch_size, c_size, m_size, labels, 2, flag=flag_c)

    A_MAE = np.zeros([4, L])
    B_MAE = np.zeros([4, L])

    for i in range(L):
        print(i)
        temp0 = model_unet_c.evaluate(train_gen, steps=steps, max_queue_size=8, workers=8, use_multiprocessing=True)
        temp1 = model_unet_c.evaluate(train_gen_LRT, steps=steps, max_queue_size=8, workers=8, use_multiprocessing=True)
        temp2 = model_unet_c.evaluate(train_gen_HRZ, steps=steps, max_queue_size=8, workers=8, use_multiprocessing=True)
        temp3 = model_unet_c.evaluate(train_gen_LRZ, steps=steps, max_queue_size=8, workers=8, use_multiprocessing=True)
        B_MAE[0, i] = temp0[3]
        B_MAE[1, i] = temp1[3]
        B_MAE[2, i] = temp2[3]
        B_MAE[3, i] = temp3[3]

        temp0 = model_unet_a.evaluate(train_gen3, steps=steps, max_queue_size=8, workers=8, use_multiprocessing=True)
        temp1 = model_unet_a.evaluate(train_gen3_LRT, steps=steps, max_queue_size=8, workers=8, use_multiprocessing=True)
        temp2 = model_unet_a.evaluate(train_gen3_HRZ, steps=steps, max_queue_size=8, workers=8, use_multiprocessing=True)
        temp3 = model_unet_a.evaluate(train_gen3_LRZ, steps=steps, max_queue_size=8, workers=8, use_multiprocessing=True)
        A_MAE[0, i] = temp0[1]
        A_MAE[1, i] = temp1[1]
        A_MAE[2, i] = temp2[1]
        A_MAE[3, i] = temp3[1]

    data = {'A_MAE':A_MAE, 'B_MAE':B_MAE}
    np.save(save_dir+'UNET_{}_PERMUTE.npy'.format(VAR), data)