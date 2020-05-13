import sys
import numpy as np
from glob import glob
import tensorflow as tf
from random import shuffle
from tensorflow import keras
import tensorflow.keras.backend as K
# utils
sys.path.insert(0, '/glade/u/home/ksha/ML_repo/utils/')
import ANN_utils as au
import data_utils as du
import keras_utils as ku
from namelist_PRISM import *

# Macros
VAR = 'TMAX'
c_size = 4; norm = 'STD'; kind = 'std_clim'; batch_num = 200
layer_N = [56, 112, 224, 448]
lr = [5e-5, 5e-6]
# data pipeline setup
L_train, batch_size, m_size = 800, 1, batch_num
steps = L_train//batch_size
labels = ['X', 'Y']; flag_c = [True]*c_size

seasons = ['djf', 'mam', 'jja', 'son']
for sea in seasons:
    print('========== {} =========='.format(sea))
    # validation batches
    valid_files = glob(BATCH_dir+'{}*BATCH*{}*VORI_{}*.npy'.format(VAR, kind, sea))
    shuffle(valid_files)
    gen_valid = ku.grid_grid_gen(valid_files, batch_size, c_size, m_size, labels, flag=flag_c)
    # model import
    save_dir = '/glade/work/ksha/data/Keras/BACKUP/'
    model_name = 'UNET_{}_B4_{}'.format(VAR, sea)
    model_name_tune = 'UNET_{}_B4_{}_tune_train'.format(VAR, sea)
    model = keras.models.load_model(save_dir+model_name+'.hdf') # restart options
    ## opt
    opt_adam = keras.optimizers.SGD(lr=5e-5, decay=0.01)
    ## callback
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_HR_temp_loss', min_delta=0.0000001, patience=3, verbose=True),
                 keras.callbacks.ModelCheckpoint(filepath=save_dir+model_name_tune+'.hdf', verbose=True,
                                                 monitor='val_HR_temp_loss', save_best_only=True)]
    model.compile(loss=keras.losses.mean_absolute_error, optimizer=opt_adam, metrics=[keras.losses.mean_absolute_error])
    train_files = glob(BATCH_dir+'{}*BATCH*{}*TORI_{}*.npy'.format(VAR, kind, sea))
    shuffle(train_files)
    gen_train = ku.grid_grid_gen(train_files, batch_size, c_size, m_size, labels, flag=flag_c)
    temp = model.fit_generator(generator=gen_train, validation_data=gen_valid, callbacks=callbacks, 
                               steps_per_epoch=steps, initial_epoch=0, epochs=100, verbose=1, shuffle=True, max_queue_size=9, workers=9)
    np.save(save_dir+model_name_tune+'.npy', temp.history)
