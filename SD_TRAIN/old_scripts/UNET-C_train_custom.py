# general tools
import sys
import argparse
from glob import glob


# data tools
import numpy as np
from random import shuffle

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

# parse user inputs
parser = argparse.ArgumentParser()
# positionals
parser.add_argument('v', help='Downscaling variable name')
parser.add_argument('s', help='Training seasons summer/winter')
parser.add_argument('c1', help='Number of input channels (1<c<5)')
parser.add_argument('c2', help='Number of output channels (1<c<3)')
args = vars(parser.parse_args())
# parser handling
VAR, seasons, input_flag, output_flag = tu.parser_handler(args)
N_input = int(np.sum(input_flag))
N_output = int(np.sum(output_flag))
if N_output > 1:
    raise ValueError('UNet accepts only one target')
    
# UNet model macros
# # hidden layer numbers (symetrical)
if VAR == 'PCT':
    print('PCT hidden layer setup')
    N = [64, 96, 128, 160]
else:
    print('T2 hidden layer setup')
    N = [56, 112, 224, 448]

l = [5e-4, 5e-5] # learning rate schedules 

# training set macros
layer_id = [7, 13, 19] # perceptual loss layers
num_train = 600 # batches per epoch
labels = ['batch', 'batch'] # input and output labels
file_path = BATCH_dir

activation='relu'
pool=False # stride convolution instead of maxpooling

def loss_model(VAR, sea, layer_id):
    model_path = temp_dir+'DAE_{}_{}_self.hdf'.format(VAR, sea) # model checkpoint
    AE = keras.models.load_model(model_path)
    W = AE.get_weights()

    N = [48, 96, 192, 384]
    input_size = (None, None, 1)
    # DAE
    DAE = mu.DAE(N, input_size)
    DAE.set_weights(W)
    # freeze
    DAE.trainable = False
    for layer in DAE.layers:
        layer.trainable = False
    f_sproj = [DAE.layers[i].output for i in layer_id]
    
    loss_models = []
    for single_proj in f_sproj:
        loss_models.append(keras.models.Model(DAE.inputs, single_proj))
    return loss_models

# LOSS function
def CLOSS(y_true, y_pred):
    '''
    MAE style content loss
    '''
    return 0.25*K.mean(K.abs(loss_models[0](y_true) - loss_models[0](y_pred)))+\
           0.25*K.mean(K.abs(loss_models[1](y_true) - loss_models[1](y_pred)))+\
           0.25*K.mean(K.abs(loss_models[2](y_true) - loss_models[2](y_pred)))+\
           0.25*K.mean(K.abs(y_true -y_pred))

for sea in seasons:
    print('===== {} training ====='.format(sea))
    # Perceptual loss definition
    loss_models = loss_model(VAR, sea, layer_id)
    
    # model name
    model_name = 'UNET-C{}_{}_{}'.format(N_input, VAR, sea)
    model_path = temp_dir+model_name+'.hdf' # model checkpoint
    train_path = temp_dir+model_name+'.npy' # train history
    # full list of files
    trainfiles = glob(file_path+'{}_BATCH_*_TORI*_{}*.npy'.format(VAR, sea)) # e.g., TMAX_BATCH_128_VORIAUG_mam30.npy
    validfiles = glob(file_path+'{}_BATCH_*_VORI_*{}*.npy'.format(VAR, sea))
    # shuffle filenames
    shuffle(trainfiles)
    shuffle(validfiles)
    
    # hist dict
    hist_total = {'loss': [], 'mean_absolute_error': [], 'val_loss': [], 'val_mean_absolute_error': []}
    
    # validation set generator
    gen_valid = tu.grid_grid_gen(validfiles, labels, input_flag, output_flag)

    # model initialization
    model = mu.UNET(N, (None, None, N_input), pool=pool, activation=activation)
    # optimizer
    opt_adam = keras.optimizers.Adam(lr=l[0])
    # callback
    callbacks = [keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=True, monitor='val_loss', save_best_only=True)]
    # compile
    model.compile(loss=CLOSS, optimizer=opt_adam, metrics=[keras.losses.mean_absolute_error])
    
    # training
    N1 = 2; N2 = 3; N3 = N1+N2
    for i in range(10):
        # counting
        N_base = i*N3; N_temp1 = N_base+N1; N_temp2 = N_base+N1+N2
        print('----- Round {} -----\nepoches: {} --> {} --> {}'.format(N_base, N_base, N_temp1, N_temp2))
        # train gen
        shuffle(trainfiles) # shuffle train files
        gen_train = tu.grid_grid_gen_sub(trainfiles[:num_train], labels, input_flag, output_flag)
        # train & learning rate swap
        K.set_value(model.optimizer.lr, l[0])
        temp_hist1 = model.fit_generator(generator=gen_train, validation_data=gen_valid, callbacks=callbacks, 
                                         steps_per_epoch=num_train, initial_epoch=N_base, epochs=N_temp1, 
                                         verbose=1, shuffle=True, max_queue_size=8, workers=8)
        K.set_value(model.optimizer.lr, l[1])
        temp_hist2 = model.fit_generator(generator=gen_train, validation_data=gen_valid, callbacks=callbacks, 
                                         steps_per_epoch=num_train, initial_epoch=N_temp1, epochs=N_temp2, 
                                         verbose=1, shuffle=True, max_queue_size=8, workers=8)
        # backup history
        hist_total = tu.dict_list_append(hist_total, temp_hist1.history)
        hist_total = tu.dict_list_append(hist_total, temp_hist2.history)
        np.save(train_path, hist_total)
