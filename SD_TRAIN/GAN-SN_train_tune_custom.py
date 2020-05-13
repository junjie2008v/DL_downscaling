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
parser.add_argument('t', help='Tuning rounds')
args = vars(parser.parse_args())
# parser handling
VAR, seasons, input_flag, output_flag = tu.parser_handler(args)
N_input = int(np.sum(input_flag))
N_output = int(np.sum(output_flag))
if N_output > 1:
    raise ValueError('UNet accepts only one target')
tune = int(args['t'])
# UNet model macros
# # hidden layer numbers (symetrical)
if VAR == 'PCT':
    print('PCT hidden layer setup')
    N = [64, 96, 128, 160]
else:
    print('T2 hidden layer setup')
    N = [48, 96, 192, 384] #[56, 112, 224, 448]

def style_fixer(source_model, style_model_path):
    backbone = keras.models.load_model(style_model_path)
    for i in range(len(backbone.layers)):
        if source_model.layers[i].name[:5] == 'dense':
            W = backbone.layers[i].get_weights()
            source_model.layers[i].set_weights(W)
            source_model.layers[i].trainable = False
    return source_model

l = [5e-5, 5e-5] # G lr; D lr
epochs = 150
lmd = 1e-3
# early stopping settings
min_del = 0
max_tol = 10 # early stopping with patience

activation='relu' # leakyReLU instead of ReLU
pool=False # stride convolution instead of maxpooling

latent_lev = 8
latent_size = 512
mapping_size = 512

# overwrite flags
input_flag = [False, True, False, False, True, True] # LR T2, HR elev, LR elev
output_flag = [True, False, False, False, False, False] # HR T2
inout_flag = [True, True, False, False, True, True]
labels = ['batch', 'batch'] # input and output labels
key = 'STYLE'



for sea in seasons:
    # ----- model name ----- #
    if tune <= 1:
        # first round of tuning
        model_path_G = temp_dir+'{}_G_TMEAN_{}.hdf'.format(key, sea)
        model_path_D = temp_dir+'{}_D_TMEAN_{}.hdf'.format(key, sea)
    else:
        # after the first round
        model_path_G = temp_dir+'{}_G_TMEAN_{}_tune{}.hdf'.format(key, sea, tune-1)
        model_path_D = temp_dir+'{}_D_TMEAN_{}_tune{}.hdf'.format(key, sea, tune-1)
    # ----- G ----- #
    input_size = (None, None, N_input)
    G_style = mu.UNET_STYLE(N, input_size, latent_lev, latent_size, mapping_size, pool=pool, activation=activation)
    opt_G = keras.optimizers.Adam(lr=0)
    print('Compiling G')
    G_style.compile(loss=keras.losses.mean_squared_error, optimizer=opt_G)
    W = tu.dummy_loader(model_path_G)
    G_style.set_weights(W)
    
    #style_model_path = temp_dir+'HALF_G_STYLE.hdf'
    #G_style = style_fixer(G_style, style_model_path)
    # ------------- #
    
    # ----- D ----- #
    input_size = (None, None, N_input+1)
    D = mu.vgg_descriminator(N, input_size)
    opt_D = keras.optimizers.Adam(lr=l[1])
    print('Compiling D')
    D.compile(loss=keras.losses.mean_squared_error, optimizer=opt_D)
    W = tu.dummy_loader(model_path_D)
    D.set_weights(W)
    D.trainable = False
    for layer in D.layers:
        layer.trainable = False
    # ------------- #
    
    # ----- GAN ----- #
    GAN_IN2 = keras.layers.Input(shape=[latent_size])
    GAN_IN3 = keras.layers.Input((None, None, N_input))

    G_OUT = G_style([GAN_IN2, GAN_IN3])
    D_IN = keras.layers.Concatenate()([G_OUT, GAN_IN3])
    D_OUT = D(D_IN)
    GAN = keras.models.Model([GAN_IN2, GAN_IN3], [G_OUT, D_OUT])
    # optimizer
    opt_GAN = keras.optimizers.Adam(lr=l[0])
    print('Compiling GAN')
    # content_loss + 1e-3 * adversarial_loss
    GAN.compile(loss=[keras.losses.mean_squared_error, keras.losses.binary_crossentropy], 
                loss_weights=[1.0, lmd],
                optimizer=opt_GAN)
    
    # ---------- Training settings ---------- #
    # Filepath
    file_path = BATCH_dir
    trainfiles = glob(file_path+'TMEAN_BATCH_*_TORI_*{}*.npy'.format(sea)) # e.g., TMAX_BATCH_128_VORIAUG_mam30.npy
    validfiles1 = glob(file_path+'TMEAN_BATCH_*_TSUB_*{}*.npy'.format(sea)) + glob(file_path+'TMEAN_BATCH_*_VSUB_*{}*.npy'.format(sea))
    validfiles2 = glob(file_path+'TMEAN_BATCH_*_VORI_*{}*.npy'.format(sea))
    #
    L_train = len(trainfiles)
    gen_valid1 = tu.grid_grid_gen_noise(validfiles1, labels, input_flag, output_flag, latent_size)
    gen_valid2 = tu.grid_grid_gen_noise(validfiles2, labels, input_flag, output_flag, latent_size)
    # model names
    G_path = temp_dir+'{}_G_TMEAN_{}_tune{}.hdf'.format(key, sea, tune)
    D_path = temp_dir+'{}_D_TMEAN_{}_tune{}.hdf'.format(key, sea, tune)
    hist_path = temp_dir+'{}_LOSS_TMEAN_{}_tune{}.npy'.format(key, sea, tune)

    # loss backup
    GAN_LOSS = np.zeros([int(epochs*L_train), 3])*np.nan
    D_LOSS = np.zeros([int(epochs*L_train)])*np.nan
    V_LOSS = np.zeros([epochs])*np.nan
    V_LOSS_TRANS = np.zeros([epochs])*np.nan
    tol = 0
    batch_size = 200
    train_size = 100
    record = 999
    for i in range(epochs):
        print('epoch = {}'.format(i))
        start_time = time.time()

        # shuffling at epoch begin
        shuffle(trainfiles)

        # loop over batches
        for j, name in enumerate(trainfiles):        

            # ----- import batch data subset ----- #
            inds = du.shuffle_ind(batch_size)[:train_size]
            temp_batch = np.load(name, allow_pickle=True)[()]
            X = temp_batch['batch'][inds, ...]
            # ------------------------------------ #

            # ----- D training ----- #
            # Latent space sampling
            Wf = np.random.normal(0.0, 1.0, size = [train_size, latent_size])
            # soft labels
            dummy_bad = np.ones(train_size)*0.05 + np.random.uniform(-0.02, 0.02, train_size)
            dummy_good = np.ones(train_size)*0.95 + np.random.uniform(-0.02, 0.02, train_size)
            # get G_output (channel last)
            g_in = [Wf, X[..., input_flag]]
            g_out = G_style.predict(g_in) # <-- np.array
            # train on batch
            d_in_fake = np.concatenate((g_out, X[..., input_flag]), axis=-1)
            d_in_true = X[..., inout_flag]
            d_loss1 = D.train_on_batch(d_in_true, dummy_good)
            d_loss2 = D.train_on_batch(d_in_fake, dummy_bad)
            d_loss = d_loss1 + d_loss2
            # ----------------------- #

            # ----- G training ----- #
            # Latent space sampling
            Wf = np.random.normal(0.0, 1.0, size = [train_size, latent_size])
            # soft labels
            dummy_good = np.ones(train_size)*0.95 + np.random.uniform(-0.02, 0.02, train_size)
            # train on batch
            gan_in = [Wf, X[..., input_flag]]
            gan_target = [X[..., output_flag], dummy_good]
            gan_loss = GAN.train_on_batch(gan_in, gan_target)
            # ---------------------- #

            # ----- Backup training loss ----- #
            D_LOSS[i*L_train+j] = d_loss
            GAN_LOSS[i*L_train+j, :] = gan_loss
            # -------------------------------- #
            if j%50 == 0:
                print('\t{} step loss = {}'.format(j, gan_loss))
        # on epoch-end
        record_temp_trans = G_style.evaluate_generator(gen_valid1, verbose=1)
        record_temp = G_style.evaluate_generator(gen_valid2, verbose=1)
        # Backup validation loss
        V_LOSS[i] = record_temp
        V_LOSS_TRANS[i] = record_temp_trans
        # Overwrite loss info
        LOSS = {'GAN_LOSS':GAN_LOSS, 'D_LOSS':D_LOSS, 
                'V_LOSS':V_LOSS, 'V_LOSS_TRANS':V_LOSS_TRANS}
        np.save(hist_path, LOSS)

        if record - record_temp > min_del:
            print('Validation loss improved from {} to {}'.format(record, record_temp))
            record = record_temp
            tol = 0
            print('tol: {}'.format(tol))
            # save
            print('save to: {}\n\t{}'.format(G_path, D_path))
            G_style.save(G_path)
            D.save(D_path)
        else:
            print('Validation loss {} NOT improved'.format(record_temp))
            tol += 1
            print('tol: {}'.format(tol))
            if tol >= max_tol:
                print('Early stopping')
                sys.exit();
            else:
                print('Pass to the next epoch')
                continue;

        print("--- %s seconds ---" % (time.time() - start_time))