# general tools
import sys
import time
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

input_flag = [False, True, True, True]
output_flag = [True, False, False, False]
    
# UNet model macros
# # hidden layer numbers (symetrical)
if VAR == 'PCT':
    print('PCT hidden layer setup')
    N = [64, 96, 128, 160]
else:
    print('T2 hidden layer setup')
    N = [56, 112, 224, 448]
    
l = 1e-5 # learning rate schedules 

# training set macros
L_train = 600 # batches per epoch
L_half = 300
labels = ['batch', 'batch'] # input and output labels
file_path = BATCH_dir

epochs = 150
min_del = 0
max_tol = 4 # early stopping with patience

for sea in seasons:
    print('===== {} training ====='.format(sea))
    record = 999
    # model name
    model_name = 'UNET{}_{}_{}_clean'.format(N_input, VAR, sea)
    model_path = temp_dir+model_name+'.hdf' # model checkpoint
    train_path = temp_dir+model_name+'_adam.npy'
    tune_path = temp_dir+model_name+'_sgd.npy'
    
    # full list of files
    validfiles = glob(file_path+'TMEAN_BATCH_*_VORI-JRA-clean_{}*'.format(sea))
    trainfile = glob(file_path+'TMEAN_BATCH_*_TORI-ERA-clean_{}*'.format(sea))
    gen_valid = tu.grid_grid_gen(validfiles[::2], labels, input_flag, output_flag)    
    # model
    model = mu.UNET(N, (None, None, N_input))
    opt_sgd = keras.optimizers.SGD(lr=l)
    model.compile(loss=keras.losses.mean_absolute_error, optimizer=opt_sgd)
    W = tu.dummy_loader(temp_dir+'UNET3_TMEAN_{}_tune.hdf'.format(sea))
    model.set_weights(W)
    
    # loss backup
    LOSS = np.zeros([int(epochs*L_train)])*np.nan
    LOSS_tune = np.zeros([int(epochs*L_train)])*np.nan
    VLOSS = np.zeros([epochs])*np.nan
    VLOSS_tune = np.zeros([epochs])*np.nan

    tol = 0
    for i in range(epochs):
        print('tune epoch = {}'.format(i))
        if i == 0:
            record = model.evaluate(gen_valid, verbose=1)
            
        start_time = time.time()
        # shuffling at epoch begin
        shuffle(trainfile)
        temp_train = trainfile[:L_train]
        shuffle(temp_train)
        
        # loop over batches
        for j, name in enumerate(temp_train):
            
            # import batch data
            temp_batch = np.load(name, allow_pickle=True)[()]
            X = temp_batch['batch'][..., input_flag]
            Y = temp_batch['batch'][..., output_flag]

            temp_loss = model.train_on_batch(X, Y)
            LOSS_tune[i*L_train+j] = temp_loss
            
            if j%50 == 0:
                print('\t{} step loss = {}'.format(j, temp_loss))
                
        # on epoch-end
        record_temp = model.evaluate(gen_valid, verbose=1)

        # Backup validation loss
        VLOSS_tune[i] = record_temp
        
        # Overwrite loss info
        LOSS_dict = {'LOSS':LOSS_tune, 'VLOSS':VLOSS_tune}
        np.save(tune_path, LOSS_dict)

        if record - record_temp > min_del:
            print('Validation loss improved from {} to {}'.format(record, record_temp))
            record = record_temp
            tol = 0
            # save
            print('save to: {}'.format(model_path))
            model.save(model_path)
        else:
            print('Validation loss {} NOT improved'.format(record_temp))
            tol += 1
            print('tol: {}'.format(tol))
            if tol >= max_tol:
                print('Early stopping')
                break;
            else:
                print('Pass to the next epoch')
                continue;
        print("--- %s seconds ---" % (time.time() - start_time))