'''
UNET tuning script. The second training stage in paper
Same augments as the first training stage, but with SGD, 
    learning rate decay and early stopping.
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

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

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
VAR, seasons, input_flag, output_flag = tu.parser_handler(args) # prefer seasons = 'annual'
N_input = int(np.sum(input_flag))  # number of input channels
N_output = int(np.sum(output_flag)) # number of (unsupervised) output channels
if N_output > 1:
    raise ValueError('UNet accepts only one target')

# # hidden layer numbers (symetrical)
if VAR == 'PCT':
    print('PCT hidden layer setup')
    N = [64, 96, 128, 160]
else:
    print('T2 hidden layer setup')
    N = [48, 96, 192, 384]

activation='relu' # leakyReLU instead of ReLU
pool=False # stride convolution instead of maxpooling
    
l = 2.5e-4 # initial learning rate

# training set macros
labels = ['batch', 'batch'] # input and output labels
file_path = BATCH_dir

for sea in seasons:
    print('===== {} tuning ====='.format(sea))
     # import pre-trained model
    model_name = 'UNET-G{}_{}_{}'.format(N_input, VAR, sea)
    model_path = temp_dir+model_name+'.hdf'
    print('Import model and weights: {}'.format(model_name))
    backbone = keras.models.load_model(model_path)
    W = backbone.get_weights()
    # tuned model
    model_name_tune = model_name+'_tune' # save separatly
    model_path_tune = temp_dir+model_name_tune+'.hdf' # checkpoint
    train_path_tune = temp_dir+model_name_tune+'.npy' # train history
    #
    model = mu.UNET(N, (None, None, N_input), pool=pool, activation=activation)
    # optimizer
    opt_sgd = keras.optimizers.SGD(lr=l, decay=0.025)
    # callbacks, tol=5
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=2, verbose=True),
                 keras.callbacks.ModelCheckpoint(filepath=model_path_tune, verbose=True,
                                                 monitor='val_loss', save_best_only=True)]
    # recompile with sgd
    model.compile(loss=keras.losses.mean_absolute_error, optimizer=opt_sgd, metrics=[keras.losses.mean_absolute_error])
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
                                       initial_epoch=0, epochs=100, verbose=1, shuffle=True, 
                                       max_queue_size=8, workers=8)
    np.save(train_path_tune, temp.history)
