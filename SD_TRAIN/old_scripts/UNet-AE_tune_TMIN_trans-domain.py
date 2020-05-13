import sys
import h5py
import numpy as np
#import pandas as pd
from glob import glob
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from random import shuffle

sys.path.insert(0, '/glade/u/home/ksha/ML_repo/utils/')
import PRISM_utils as pu
import data_utils as du
import keras_utils as ku
import ANN_utils as au
from namelist_PRISM import *
# ===== functions ===== #
def freeze_unet(model, l, lr=5e-6):
    for i, layer in enumerate(model.layers):
        if i <= l:
            layer.trainable = False
    opt_tune = keras.optimizers.Adam(lr=lr)
    model.compile(loss=keras.losses.mean_absolute_error, optimizer=opt_tune, metrics=[keras.losses.mean_absolute_error])
    return model

def freeze_unet2(model, l, lr=5e-6):
    for i, layer in enumerate(model.layers):
        if i <= l:
            layer.trainable = False
    opt_tune = keras.optimizers.Adam(lr=lr)
    model.compile(loss=[keras.losses.mean_absolute_error, keras.losses.mean_absolute_error], 
                  loss_weights=[1, 1], optimizer=opt_tune, metrics=[keras.losses.mean_absolute_error])
    return model
# ===== macros ===== #
VAR= 'TMIN'; kind='std_clim' # <--- TMAX/TMIN
seasons = ['djf', 'mam', 'jja', 'son']
layer_N = [56, 112, 224, 448] # UNet channel numbers by layer
L_train, L_etopo, batch_size, m_size = 200, 800, 1, 200
steps = L_train//batch_size
steps_etopo = L_etopo//batch_size

cut, cut_etopo = 100, 300; 
map_sizes = [64, 96, 128]
c_size = 4
labels = ['X', 'Y']; flag_c = [True]*c_size
save_dir = '/glade/work/ksha/data/Keras/BACKUP/'
for sea in seasons:
    print(sea)
    # validation data
    valid_sub = glob(BATCH_dir+'{}*BATCH*{}*SUB_{}*.npy'.format(VAR, kind, sea))
    gen_valid_sub = ku.grid_grid_gen(valid_sub, batch_size, c_size, m_size, labels, flag=flag_c)
    # --------------------------------------------------------------------------------------- #
    # ===== step 1: elev. tune ===== #
    record_t = 120; max_tol = 5; tol = 0
    for n in range(20):
        print('====================')
        if n == 0:
            model_name = 'UNET_{}_B4_{}.hdf'.format(VAR, sea)
            unet_backbone = keras.models.load_model(save_dir+model_name)
            unet_tune2 = au.UNET_AE(layer_N, c_size, dropout=False)
            unet_tune2 = freeze_unet2(unet_tune2, l=0, lr=1e-6)
            unet_tune2.set_weights(unet_backbone.get_weights())
            # UNet without HR temp branch
            unet_tune = au.UNET(layer_N, c_size)
            unet_tune = freeze_unet(unet_tune, l=19, lr=5e-6) # 29
        # weights initialization #
        W_AE = unet_tune2.get_weights()
        W_ZH = W_AE[0:-14]+[W_AE[-13]]+W_AE[-8 :-4]+W_AE[-2:]
        unet_tune.set_weights(W_ZH)
        # ========== data pipeline setup ========== # 
        # training
        record = 999
        for i in range(1):
            train_files = []
            for map_size in map_sizes:
                # multi-scale training (with augmented batches prepared)
                temp_tune_etopo = glob(BATCH_dir+'{}*BATCH*{}*{}*TMIX_{}*.npy'.format(VAR, kind, map_size, sea)) # trans + tune
                temp_ori_etopo = glob(BATCH_dir+'{}*BATCH*{}*{}*TORI_{}*.npy'.format(VAR, kind, map_size, sea)) # train
                shuffle(temp_tune_etopo); shuffle(temp_ori_etopo) # shuffle before training
                train_files += temp_tune_etopo[:cut_etopo]
                train_files += temp_ori_etopo[:int(0.1*cut_etopo)]
            train_subset = train_files[:L_etopo]
            gen_train_etopo = ku.grid_grid_genZ(train_subset, batch_size, c_size, m_size, labels, flag=flag_c)
            unet_tune.fit_generator(generator=gen_train_etopo, steps_per_epoch=steps_etopo, 
                                    epochs=1, verbose=1, shuffle=True, max_queue_size=8, workers=8)
        # --------------------------------------------------------------------------------------- #
        # ===== step 2: HR temp adjust ===== #
        # initialize weights
        W_etopo = unet_tune.get_weights()
        tail_ZH = W_etopo[-7:]
        tail_T2 = np.copy([W_AE[-14]]+W_AE[-12:-8]+W_AE[-4:-2])
        # HR temp branch
        W_AE[:-13] = W_etopo[:-6]
        W_AE[-14] = tail_T2[0]
        W_AE[-12:-8] = tail_T2[1:5]
        W_AE[-4:-2] = tail_T2[5:7]
        # HR elev branch
        W_AE[-13] = tail_ZH[0]
        W_AE[-8:-4] = tail_ZH[1:5]
        W_AE[-2:] = tail_ZH[5:7]
        #
        unet_tune2.set_weights(W_AE)
        # ============= train T2 ============ #
        train_files = []
        for map_size in map_sizes:
            temp_list = glob(BATCH_dir+'{}*BATCH*{}*{}*TMIX_{}*.npy'.format(VAR, kind, map_size, sea))
            shuffle(temp_list) # <--- shuffle before training
            train_files += temp_list[:cut]
        shuffle(train_files)
        train_subset = train_files[:L_train]
        gen_train = ku.grid_grid_gen(train_subset, batch_size, c_size, m_size, labels, flag=flag_c)
        # train
        hist = unet_tune2.fit_generator(generator=gen_train, validation_data=gen_valid_sub,
                                 steps_per_epoch=steps, epochs=1, verbose=1, shuffle=True, max_queue_size=8, workers=8)
        # mannual early stopping (by HR temp MAEs)
        print('val_HR_temp_loss: {}'.format(hist.history['val_HR_temp_loss'][0]))
        print('tol: {}'.format(tol))
        if hist.history['val_HR_temp_loss'][0] < record_t:
            print('Hit')
            record_t = hist.history['val_HR_temp_loss']
            unet_tune2.save(save_dir+'UNET_{}_B4_{}_tune_trans.hdf'.format(VAR, sea)) # save step 2 weights
        elif tol < max_tol:
            print('Pass')
            tol += 1
        else:
            print('Early stopping')
            break;


