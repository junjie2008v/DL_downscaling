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
VAR = 'PCT'
c_size = 4; norm = 'LIN'; kind = 'pct'; batch_num = 200
layer_N = [38, 68, 98, 128]
lr = [2.5e-4, 2.5e-5]
seasons = ['mam']
BATCH_dir = BATCH_dir+'temp_batches/'

# data pipeline setup
L_train, batch_size, m_size = 600, 1, batch_num
steps = L_train//batch_size
labels = ['X', 'Y']; flag_c = [True]*c_size
for sea in seasons:
    # validation batches
    valid_dir = BATCH_dir+'{}*BATCH*{}*VORI*{}*.npy'.format(VAR, kind, sea)
    valid_files = glob(valid_dir)
    valid_files = valid_files[::2]
    gen_valid = ku.grid_grid_gen3(valid_files, batch_size, c_size, m_size, labels, flag=flag_c)
    # hist dict
    hist_total = {'loss': [], 'mean_absolute_error': [], 'val_loss': [], 'val_mean_absolute_error': []}
    # model definition
    save_dir = '/glade/work/ksha/data/Keras/BACKUP/'
    model_name = 'XNET4_{}_{}'.format(VAR, sea)
    # input end
    IN = keras.layers.Input((None, None, c_size))
    X11_conv = au.input_conv(IN, layer_N[0])
    # down-sampling blocks
    X21_conv = au.down_block(X11_conv, layer_N[1])
    X31_conv = au.down_block(X21_conv, layer_N[2])
    X41_conv = au.down_block(X31_conv, layer_N[3])
    # up-sampling blocks 2
    X12_conv = au.up_block(X21_conv, [X11_conv], layer_N[0]) 
    X22_conv = au.up_block(X31_conv, [X21_conv], layer_N[1])
    X32_conv = au.up_block(X41_conv, [X31_conv], layer_N[2])
    # up-sampling blocks 3
    X13_conv = au.up_block(X22_conv, [X11_conv, X12_conv], layer_N[0]) 
    X23_conv = au.up_block(X32_conv, [X21_conv, X22_conv], layer_N[1]) 
    # up-sampling blocks 4
    X14_conv = au.up_block(X23_conv, [X11_conv, X12_conv, X13_conv], layer_N[0]) 
    # output end
    OUT = keras.layers.Conv2D(1, 1, activation=keras.activations.linear)(X14_conv)
    model = keras.models.Model(inputs=[IN], outputs=[OUT])
    ## opt
    opt_adam = keras.optimizers.Adam(lr=lr[0])
    ## callback
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0000001, patience=5, verbose=True),
                 keras.callbacks.ModelCheckpoint(filepath=save_dir+model_name+'.hdf', verbose=True,
                                                 monitor='val_loss', save_best_only=True)]
    model.compile(loss=keras.losses.mean_absolute_error, optimizer=opt_adam, metrics=[keras.losses.mean_absolute_error])
    ## training loop (100 epoches)
    N1 = 2; N2 = 3; N3 = N1+N2
    cut = 210; map_sizes = [64, 96, 128]
    for i in range(15):
        train_files = []
        for map_size in map_sizes:
            temp_list = glob(BATCH_dir+'{}*BATCH*{}*{}*TORI*{}*.npy'.format(VAR, kind, map_size, sea))
            shuffle(temp_list)
            train_files += temp_list[:cut]
        shuffle(train_files); 
        train_subset = train_files[:L_train]
        print('===== shuffle data =====')
        # train gen
        N_base = i*N3; N_temp1 = N_base+N1; N_temp2 = N_base+N1+N2
        print('\tepoches: {}:{}:{}'.format(N_base, N_temp1, N_temp2))
        K.set_value(model.optimizer.lr, lr[0])
        gen_train = ku.grid_grid_gen3(train_subset, batch_size, c_size, m_size, labels, flag=flag_c)
        # train
        temp_hist1 = model.fit_generator(generator=gen_train, validation_data=gen_valid, callbacks=callbacks, 
                                         steps_per_epoch=steps, initial_epoch=N_base, epochs=N_temp1, 
                                         verbose=1, shuffle=True, max_queue_size=9, workers=9)
        K.set_value(model.optimizer.lr, lr[1]) # <--- learning rate swap
        temp_hist2 = model.fit_generator(generator=gen_train, validation_data=gen_valid, callbacks=callbacks, 
                                         steps_per_epoch=steps, initial_epoch=N_temp1, epochs=N_temp2, 
                                         verbose=1, shuffle=True, max_queue_size=9, workers=9)
        # backup hist
        hist_total = du.dict_list_append(hist_total, temp_hist1.history)
        hist_total = du.dict_list_append(hist_total, temp_hist2.history)
        np.save(save_dir+model_name+'.npy', hist_total)

