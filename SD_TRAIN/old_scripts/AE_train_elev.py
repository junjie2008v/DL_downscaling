# general tools
import sys
import argparse
from glob import glob

# data tools
import numpy as np

# deep learning tools
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/')
from namelist import *
import model_utils as mu
import train_utils as tu

VAR = 'TMEAN'; sea = 'jja'
# flags
input_flag = [False, False, False, False, True, False] # HR target and elev
output_flag = [False, False, False, False, True, False]
labels = ['batch', 'batch']

# training settings
l = [1e-4, 1e-5] # learning rate
epochs = 200 # sapce for early stopping

# DAE
N = [48, 96, 192, 384]
input_size = (None, None, 1)

# training file location
file_path = BATCH_dir
trainfiles = glob(file_path+'{}_BATCH_*_TORI*_{}*.npy'.format(VAR, sea))
validfiles = glob(file_path+'{}_BATCH_*_VORI*_{}*.npy'.format(VAR, sea))
#
model_path = temp_dir+'DAE_{}_{}_elev.hdf'.format(VAR, sea)
train_path = temp_dir+'DAE_{}_{}_elev.npy'.format(VAR, sea)

DAE = mu.DAE(N, input_size)

# optimizer & callback & compile
opt_ae = keras.optimizers.Adam(lr=l[0])
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=2, verbose=True),
             keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=True, monitor='val_loss', save_best_only=True)]
DAE.compile(loss=keras.losses.mean_absolute_error, optimizer=opt_ae, metrics=[keras.losses.mean_absolute_error])

# Data generator
gen_train = tu.grid_grid_gen(trainfiles, labels, input_flag, output_flag)
gen_valid = tu.grid_grid_gen(validfiles, labels, input_flag, output_flag)

# train
temp_hist = DAE.fit_generator(generator=gen_train, validation_data=gen_valid, callbacks=callbacks, 
                              initial_epoch=0, epochs=epochs, verbose=1, shuffle=True, max_queue_size=8, workers=8)

W = DAE.get_weights() # backup weights
DAE_tune = mu.DAE(N, input_size)
opt_ae = keras.optimizers.SGD(lr=l[1], decay=1e-2*l[1])

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0000001, patience=2, verbose=True),
             keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=True, monitor='val_loss', save_best_only=True)]

DAE_tune.compile(loss=keras.losses.mean_absolute_error, optimizer=opt_ae, metrics=[keras.losses.mean_absolute_error])
DAE_tune.set_weights(W)

temp_hist = DAE_tune.fit_generator(generator=gen_train, validation_data=gen_valid, callbacks=callbacks, 
                              initial_epoch=0, epochs=epochs, verbose=1, shuffle=True, max_queue_size=8, workers=8)
