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

l = 2.5e-4 # initial learning rate 

# training set macros
num_train = 600 # batches per epoch
labels = ['batch', 'batch'] # input and output labels
file_path = BATCH_dir + 'temp_batches/'

# AE settings
AE_N = [32, 32, 32, 32]
AE_input_size = (None, None, 1)
AE_latent_size = (None, None, AE_N[-1])
# Selected AE layers
layer_id = [5, 9, 13] #5, 9, 13

# LOSS function

def dummy_loss_model(VAR, sea):
    AE_path = temp_dir+'DAE_{}_{}.hdf'.format(VAR, sea) # model checkpoint
    AE = keras.models.load_model(AE_path)
    W = AE.get_weights()
    DAE = mu.DAE(AE_N, AE_input_size, AE_latent_size)
    DAE.set_weights(W)
    # Encoder layer selection
    encoder = DAE.layers[1]
    # freeze enoder layers
    encoder.trainable = False
    for layer in encoder.layers:
        layer.trainable = False
    f_sproj = [encoder.layers[i].output for i in layer_id]
    # Loss models
    loss_models = []
    for single_proj in f_sproj:
        loss_models.append(keras.models.Model(encoder.inputs, single_proj))
    return loss_models

def dummy_model_loader(N_input, VAR, sea):
    # model name
    model_name = 'UNET-C{}_{}_{}'.format(N_input, VAR, sea)
    model_path = temp_dir+model_name+'.hdf'
    # pre-trained weights
    model = keras.models.load_model(model_path, compile=False)
    W = model.get_weights()
    return W

def CLOSS(y_true, y_pred):
    '''
    MAE style content loss
    '''
    return 0.33*K.mean(K.abs(loss_models[0](y_true) - loss_models[0](y_pred)))+\
           0.33*K.mean(K.abs(loss_models[1](y_true) - loss_models[1](y_pred)))+\
           0.34*K.mean(K.abs(loss_models[2](y_true) - loss_models[2](y_pred)))

#     return 0.25*K.mean(K.abs(loss_models[0](y_true) - loss_models[0](y_pred)))+\
#            0.25*K.mean(K.abs(loss_models[1](y_true) - loss_models[1](y_pred)))+\
#            0.25*K.mean(K.abs(loss_models[2](y_true) - loss_models[2](y_pred)))+\
#            0.25*K.mean(K.abs(y_true -y_pred))

for sea in seasons:
    print('===== {} training ====='.format(sea))
    # Content loss definition
    loss_models = dummy_loss_model(VAR, sea)
    W = dummy_model_loader(N_input, VAR, sea)

    # tuned model
    model_name_tune = 'UNET-C{}_{}_{}_tune'.format(N_input, VAR, sea)
    model_path_tune = temp_dir+model_name_tune+'.hdf' # checkpoint
    train_path_tune = temp_dir+model_name_tune+'.npy' # train history
    model = mu.UNET(N, (None, None, N_input))
    # optimizer
    opt_sgd = keras.optimizers.SGD(lr=l, decay=0.025)
    # callback
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=2, verbose=True),
                 keras.callbacks.ModelCheckpoint(filepath=model_path_tune, verbose=True,
                                                 monitor='val_loss', save_best_only=True)]
    # compile
    model.compile(loss=CLOSS, optimizer=opt_sgd, metrics=[keras.losses.mean_absolute_error])
    model.set_weights(W)

    # full list of training files
    trainfiles = glob(file_path+'{}_BATCH_*_TORI_*{}*.npy'.format(VAR, sea)) # e.g., TMAX_BATCH_128_VORI_mam30.npy, 
    validfiles = glob(file_path+'{}_BATCH_*_VORI_*{}*.npy'.format(VAR, sea)) # excluding "ORIAUG" pattern
    # shuffle filenames
    shuffle(trainfiles)
    shuffle(validfiles)
    # generators
    gen_valid = tu.grid_grid_gen(validfiles, labels, input_flag, output_flag)
    gen_train = tu.grid_grid_gen(trainfiles, labels, input_flag, output_flag)
    # tuning
    temp = model.fit_generator(generator=gen_train, validation_data=gen_valid, callbacks=callbacks, 
                                       initial_epoch=0, epochs=50, verbose=1, shuffle=True, 
                                       max_queue_size=8, workers=8)
    np.save(train_path_tune, temp.history)
