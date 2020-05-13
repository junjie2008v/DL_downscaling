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
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/')
from namelist import *
import data_utils as du
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
    N = [48, 96, 192, 384] #[56, 112, 224, 448]


l = [1e-4, 2e-4] # l[0] is not used (G is not trained)
batch_size = 200
epochs = 30 # fixed 5 epochs for pre-trainined 
mix = True # <--- mix true and fake samples? False means train real --> train fake 
pool = False

input_flag = [False, True, False, False, True, True] # LR T2, HR elev, LR elev
output_flag = [True, False, False, False, False, False] # HR T2
inout_flag = [True, True, False, False, True, True]
labels = ['batch', 'batch'] # input and output labels

file_path = BATCH_dir

for sea in seasons:
    print('========== {} =========='.format(sea))
    # train with tuned UNET
    model_name = 'UNET-G{}_{}_{}_tune'.format(N_input, VAR, sea)
    model_path = temp_dir+model_name+'.hdf'
    print('Import model: {}'.format(model_name))
    backbone = keras.models.load_model(model_path)
    W = backbone.get_weights()

    # generator
    G = mu.UNET(N, (None, None, N_input), pool=pool)
    # optimizer
    opt_G = keras.optimizers.Adam(lr=l[0])

    print('Compiling G')
    G.compile(loss=keras.losses.mean_absolute_error, optimizer=opt_G)
    G.set_weights(W)
    
    input_size = (None, None, N_input+1)
    D = mu.vgg_descriminator(N, input_size)

    opt_D = keras.optimizers.Adam(lr=l[1])
    print('Compiling D')
    D.compile(loss=keras.losses.mean_squared_error, optimizer=opt_D)
    

    trainfiles = glob(file_path+'{}_BATCH_*_TORI_*{}*.npy'.format(VAR, sea)) # e.g., TMEAN_BATCH_128_TORI_mam30.npy
    model_path = temp_dir+'NEO_D_{}_{}_pretrain.hdf'.format(VAR, sea)
    hist_path = temp_dir+'NEO_D_{}_{}_pretrain.npy'.format(VAR, sea)
    L_train = len(trainfiles)
    D_LOSS = np.zeros([int(epochs*L_train)])*np.nan
    
    # training epochs
    for i in range(epochs):
        print('epoch = {}'.format(i))
        shuffle(trainfiles)
        for j, name in enumerate(trainfiles):
            # soft labels with MSE
            dummy_bad = np.ones(batch_size)*0.1 + np.random.uniform(-0.02, 0.02, batch_size)
            dummy_good = np.ones(batch_size) - dummy_bad
            
            # import batch data
            temp_batch = np.load(name, allow_pickle=True)[()]
            X = temp_batch['batch']

            # D training
            D.trainable = True
            g_in = X[..., input_flag]
            g_out = G.predict([g_in]) # <-- np.array

            d_in_fake = np.concatenate((g_out, g_in), axis=-1) # channel last
            d_in_true = X[..., inout_flag]
            if mix:
                d_in = np.concatenate((d_in_fake, d_in_true), axis=0) # batch size doubled
                d_target = np.concatenate((dummy_bad, dummy_good), axis=0)
                d_shuffle_ind = du.shuffle_ind(2*batch_size)
                d_loss1 = D.train_on_batch(d_in[d_shuffle_ind, ...], d_target[d_shuffle_ind, ...])
                d_loss2 = 0
            else:
                d_loss1 = D.train_on_batch(d_in_true, dummy_good)
                d_loss2 = D.train_on_batch(d_in_fake, dummy_bad)
                
            d_loss_sum = d_loss1 + d_loss2
            D_LOSS[i*L_train+j] = d_loss_sum
            LOSS = {'D_LOSS':D_LOSS}
            np.save(hist_path, LOSS)

            if j%50 == 0:
                print('\t{} step loss = {}'.format(j, d_loss_sum))
        print('Save to {}'.format(model_path))
        D.save(model_path)