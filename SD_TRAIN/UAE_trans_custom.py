'''
UNET tuning script. Transferring from one domain to another
'''

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

def freeze_unet(model, l, lr=1e-6):
    for i, layer in enumerate(model.layers):
        if i <= l:
            layer.trainable = False
    opt_tune = keras.optimizers.Adam(lr=lr)
    model.compile(loss=keras.losses.mean_absolute_error, optimizer=opt_tune, metrics=[keras.losses.mean_absolute_error])
    return model

def freeze_uae(model, l, lr=1e-6):
    for i, layer in enumerate(model.layers):
        if i <= l:
            layer.trainable = False
    opt_tune = keras.optimizers.SGD(lr=lr)
    model.compile(loss=[keras.losses.mean_absolute_error, keras.losses.mean_absolute_error], 
                  loss_weights=[1, 1e-5], 
                  optimizer=opt_tune, 
                  metrics=[keras.losses.mean_absolute_error])
    return model

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
if N_output <= 1:
    raise ValueError('UNet-AE accepts more than one target')
N_output = N_output - 1 # -1 means number of unsupervised target only

N = [56, 112, 224, 448]
l = 2.5e-4 # initial learning rate

# training set macros
labels = ['batch', 'batch'] # input and output labels
file_path = BATCH_dir + 'temp_batches/'
out_flag_elev = [False, False, False, False, True, False] # pointing HR elev
min_del = 0 # <----- !
max_iter = 12

for sea in seasons:
    print('===== {} transferring ====='.format(sea))
    # params
    loss_credit = 999
    current_tol = 0
    max_tol = 2
    
    print('Preparing data generators')
    trainfiles_t2 = glob(file_path+'{}_BATCH_*_TORI_*{}*.npy'.format(VAR, sea))+\
                    glob(file_path+'{}_BATCH_*_VORI_*{}*.npy'.format(VAR, sea)) # training domain t2 target
    validfiles_t2 = glob(file_path+'{}_BATCH_*_TSUB_*{}*.npy'.format(VAR, sea))+\
                    glob(file_path+'{}_BATCH_*_VSUB_*{}*.npy'.format(VAR, sea))
    trainfiles_elev = glob(file_path+'{}_BATCH_*_TMIX_*{}*.npy'.format(VAR, sea))+\
                      glob(file_path+'{}_BATCH_*_VMIX_*{}*.npy'.format(VAR, sea)) # transferring & tuning domain elev
    # shuffle filenames
    shuffle(trainfiles_t2)
    shuffle(validfiles_t2)
    shuffle(trainfiles_elev)
    # the data generator for elev tuner and t2 tuner
    gen_elev = tu.grid_grid_gen(trainfiles_elev, labels, input_flag, out_flag_elev, sign_flag=True)
    gen_train = tu.grid_grid_gen_multi(trainfiles_t2, labels, input_flag, output_flag)
    gen_valid = tu.grid_grid_gen_multi(validfiles_t2, labels, input_flag, output_flag)
    
    print('Importing pre-trained weights')
    # import pre-trained model (e.g., 'UNET_TMAX_A3_djf.hdf')
    model_name = 'UAE{}_{}_{}_tune'.format(N_input, VAR, sea)
    model_path = temp_dir+model_name+'.hdf'
    print('\tmodel: {}'.format(model_name))
    backbone = keras.models.load_model(model_path)
    W = backbone.get_weights()
    # tuned model
    model_name_tune = 'UAE{}_{}_{}_trans'.format(N_input, VAR, sea) # save separatly
    model_path_tune = temp_dir+model_name_tune+'.hdf' # checkpoint
    
    print('Tuning model configurations')
    # elev output branch
    elev_tuner = mu.UNET(N, (None, None, N_input))
    elev_tuner = freeze_unet(elev_tuner, l=19, lr=1e-6) # 19
    # two output branches
    target_tuner = mu.UNET_AE(N, (None, None, N_input), output_channel_num=N_output, drop_rate=0)
    target_tuner = freeze_uae(target_tuner, l=0, lr=1e-6)
    
    target_tuner.set_weights(W)
    for n in range(max_iter):
        print('Tuning epoch = {}'.format(n))
        if n == 0:
            target_tuner.set_weights(W) # set backbone weights
            print('\t Performance before tuning')
            baseline_hist = target_tuner.evaluate_generator(gen_valid, verbose=1)
            print('***** Baseline val loss: {} *****'.format(baseline_hist[1]))
        
        print('\tUpdating elev_tuner weights')
        W_UAE = target_tuner.get_weights()
        W_elev = W_UAE[0:-14]+[W_UAE[-13]]+W_UAE[-8 :-4]+W_UAE[-2:]
        elev_tuner.set_weights(W_elev)
        
        print('\tUnsupervised loss opt ...')
        elev_tuner.fit_generator(generator=gen_elev, epochs=1, verbose=1, shuffle=True, max_queue_size=8, workers=8)

        print('\tUpdating weights')
        W_elev = elev_tuner.get_weights() # <------ "W_elev updated"
        tail_elev = W_elev[-7:]
        tail_T2 = np.copy([W_UAE[-14]]+W_UAE[-12:-8]+W_UAE[-4:-2])
        # encoder-decoders
        W_UAE[:-13] = W_elev[:-6]
        #
        W_UAE[-14] = tail_T2[0]
        W_UAE[-12:-8] = tail_T2[1:5]
        W_UAE[-4:-2] = tail_T2[5:7]
        # HR elev branch
        W_UAE[-13] = tail_elev[0]
        W_UAE[-8:-4] = tail_elev[1:5]
        W_UAE[-2:] = tail_elev[5:7]
        #
        target_tuner.set_weights(W_UAE)
        
        print('\tSupervised loss opt ...')
        temp_hist = target_tuner.fit_generator(generator=gen_train, validation_data=gen_valid, epochs=2, verbose=1, shuffle=True, max_queue_size=8, workers=8)                
        # (mannual) early stoppings
        val_HR_temp_loss = temp_hist.history['val_HR_temp_loss'][0]
        print('\tval_HR_temp_loss: {}; current_tol: {}'.format(val_HR_temp_loss, current_tol))
        if (loss_credit - val_HR_temp_loss) > min_del:
            loss_credit = val_HR_temp_loss
            print('\tModel checkpoint: {}'.format(model_path_tune))
            #target_tuner.save(model_path_tune)
        else:
            current_tol += 1
            print('\tNo performance gain')
            if current_tol < max_tol:
                print('\tPass')
            else:
                print('\tEarly stopping')
                break;