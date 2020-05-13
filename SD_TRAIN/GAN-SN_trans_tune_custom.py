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

def mapping_network_fixer(style_model, style_gan_name, trainable=False):
    '''
    Assuming layers names (and number of nodes) match
    '''
    W = tu.dummy_loader(style_gan_name)
    model = mu.UNET_STYLE(N, input_size, latent_lev, latent_size, mapping_size, 
                          pool=pool, activation=activation, noise=[0.2, 0.1])
    model.set_weights(W)
    W_mapping_network = []
    for layer in model.layers:
        if layer.name[:3] == 'map':
            W_mapping_network += layer.get_weights()
    if ~trainable:
        style_model.trainable = False
        for layer in style_model.layers:
            layer.trainable = False
    opt_G = keras.optimizers.Adam(lr=0)
    style_model.compile(loss=keras.losses.mean_squared_error, optimizer=opt_G)
    style_model.set_weights(W_mapping_network)
    return style_model

def mix_network_fixer(dscale_model, dscale_gan_name):
    W = tu.dummy_loader(dscale_gan_name)
    model = mu.UNET_STYLE(N, input_size, latent_lev, latent_size, mapping_size, 
                          pool=pool, activation=activation, noise=[False, False])
    model.set_weights(W)
    for layer in dscale_model.layers:
        if layer.name[:9] == 'unet_left':
            layer.trainable = False
    # compile
    opt_G = keras.optimizers.Adam(lr=0)
    dscale_model.compile(loss=keras.losses.mean_squared_error, optimizer=opt_G)
    # weights
    for layer in model.layers:
        for layer2 in dscale_model.layers:
            if layer.name == layer2.name and layer.name[:9] == 'unet_left':
                layer2.set_weights(layer.get_weights())
    return dscale_model

class grid_grid_gen_noise_temp(keras.utils.Sequence):
    def __init__(self, filename, labels, input_flag, output_flag, latent_size):

        self.filename = filename
        self.labels = labels
        self.input_flag = input_flag
        self.output_flag = output_flag
        self.latent_size = latent_size
        self.filenum = len(self.filename)
    def __len__(self):
        return self.filenum
    
    def __getitem__(self, index):
        temp_name = self.filename[index]
        return self.__readfile__(temp_name, self.labels, self.input_flag, self.output_flag, self.latent_size)
    
    def __readfile__(self, temp_name, labels, input_flag, output_flag, latent_size):
        
        data_temp = np.load(temp_name, allow_pickle=True)
        X = data_temp[()][labels[0]][..., input_flag] # channel last
        Y = data_temp[()][labels[1]][..., output_flag]
        train_size = len(Y)
        Z1 = np.random.normal(0.0, 1.0, size = [train_size, latent_size])
        Z2 = np.random.normal(0.0, 1.0, size = [train_size, latent_size])
        noise = [style_model_base.predict(Z1), style_model_elev.predict(Z2)]
        return noise+[X], [Y]

l = [1e-4, 1e-4] # G lr; D lr
epochs = 150
lmd = 1e-3
# early stopping settings
min_del = 0
max_tol = 4 # early stopping with patience

activation='leaky' # leakyReLU instead of ReLU
pool=False # stride convolution instead of maxpooling
input_size = (None, None, 3)
latent_lev = 4
latent_size = N[-1]
mapping_size = N[-1]

# overwrite flags
input_flag = [False, True, False, False, True, True] # LR T2, HR elev, LR elev
output_flag = [True, False, False, False, False, False] # HR T2
inout_flag = [True, True, False, False, True, True]
labels = ['batch', 'batch'] # input and output labels
key = 'SGAN'
model_import_dir = temp_dir
file_path = BATCH_dir

for sea in seasons:
    print('Initializing mapping network')
    style_model_base = mu.STYLE_MAP(latent_lev, latent_size, mapping_size, activation='leaky')
    style_model_base = mapping_network_fixer(style_model_base, model_import_dir+'{}_G_TMEAN_jja.hdf'.format(key), trainable=False)

    style_model_elev = mu.STYLE_MAP(latent_lev, latent_size, mapping_size, activation='leaky')
    style_model_elev = mapping_network_fixer(style_model_elev, model_import_dir+'{}_G_ELEV_jja.hdf'.format(key), trainable=False)
    
    print('Initializing the generator')
    model_mix = mu.UNET_MAP_MIX(N, input_size, latent_lev, latent_size, mapping_size, 
                                pool=pool, activation=activation, noise=[0.2, 0.1])
    model_mix = mix_network_fixer(model_mix, model_import_dir+'{}_G_TMEAN_{}.hdf'.format(key, sea))
    
    print('Initializing the discriminator')
    input_size_d = (None, None, N_input+1)
    D = mu.vgg_descriminator(N, input_size_d)
    opt_D = keras.optimizers.Adam(lr=l[1])
    print('Compiling D')
    D.compile(loss=keras.losses.mean_squared_error, optimizer=opt_D)
    D.trainable = False
    for layer in D.layers:
        layer.trainable = False
        
    print('Building GAN')
    GAN_IN1 = keras.layers.Input(shape=[latent_size])
    GAN_IN2 = keras.layers.Input(shape=[latent_size])
    GAN_IN3 = keras.layers.Input((None, None, N_input))

    W1 = style_model_base(GAN_IN1)
    W2 = style_model_elev(GAN_IN2)

    G_OUT = model_mix([W1, W2, GAN_IN3])
    D_IN = keras.layers.Concatenate()([G_OUT, GAN_IN3])
    D_OUT = D(D_IN)
    GAN = keras.models.Model([GAN_IN1, GAN_IN2, GAN_IN3], [G_OUT, D_OUT])
    # optimizer
    opt_GAN = keras.optimizers.Adam(lr=l[0])
    print('Compiling GAN')
    # content_loss + 1e-3 * adversarial_loss
    GAN.compile(loss=[keras.losses.mean_squared_error, keras.losses.binary_crossentropy], 
                loss_weights=[1.0, lmd],
                optimizer=opt_GAN)
    # Filepath
    trainfile64 = glob(file_path+'TMEAN_BATCH_64_TORI_*{}*.npy'.format(sea))
    trainfile96 = glob(file_path+'TMEAN_BATCH_96_TORI_*{}*.npy'.format(sea))
    validfiles1 = glob(file_path+'TMEAN_BATCH_64_TSUB_*{}*.npy'.format(sea)) + glob(file_path+'TMEAN_BATCH_96_TSUB_*{}*.npy'.format(sea))
    validfiles2 = glob(file_path+'TMEAN_BATCH_64_VORI_*{}*.npy'.format(sea)) + glob(file_path+'TMEAN_BATCH_96_VORI_*{}*.npy'.format(sea)) 
    #
    L_train = 320
    gen_valid1 = grid_grid_gen_noise_temp(validfiles1, labels, input_flag, output_flag, latent_size)
    gen_valid2 = grid_grid_gen_noise_temp(validfiles2, labels, input_flag, output_flag, latent_size)
    # model names
    G_name = '{}_G_TMEAN_{}_trans'.format(key, sea)
    D_name = '{}_D_TMEAN_{}_trans'.format(key, sea)
    G_path = temp_dir+G_name+'.hdf'
    D_path = temp_dir+D_name+'.hdf'
    hist_path = temp_dir+'{}_LOSS_TMEAN_{}_trans.npy'.format(key, sea)

#     gen_train = grid_grid_gen_noise_temp(trainfile64+trainfile96, labels, input_flag, output_flag, latent_size)
#     gen_valid = grid_grid_gen_noise_temp(validfiles1, labels, input_flag, output_flag, latent_size)
#     callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=True),
#                  keras.callbacks.ModelCheckpoint(filepath=G_path, verbose=True, monitor='val_loss', save_best_only=True)]
#     model_mix.fit_generator(generator=gen_train, validation_data=gen_valid, callbacks=callbacks, epochs=150,
#                             verbose=1, shuffle=True, max_queue_size=8, workers=8)
    
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
        shuffle(trainfile64)
        shuffle(trainfile96)
        trainfiles = trainfile64[:160] + trainfile96[:160]
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
            W1 = np.random.normal(0.0, 1.0, size = [train_size, latent_size])
            W2 = np.random.normal(0.0, 1.0, size = [train_size, latent_size])
            # soft labels
            dummy_bad = np.ones(train_size)*0.1 + np.random.uniform(-0.02, 0.02, train_size)
            dummy_good = np.ones(train_size)*0.9 + np.random.uniform(-0.02, 0.02, train_size)
            # get G_output (channel last)
            g_in = [W1, W2, X[..., input_flag]]
            g_out = model_mix.predict(g_in) # <-- np.array
            # train on batch
            d_in_fake = np.concatenate((g_out, X[..., input_flag]), axis=-1)
            d_in_true = X[..., inout_flag]
            d_losst1 = D.train_on_batch(d_in_true, dummy_good)
            d_lossf1 = D.train_on_batch(d_in_fake, dummy_bad)
            d_losst2 = D.train_on_batch(d_in_true, dummy_good)
            d_lossf2 = D.train_on_batch(d_in_fake, dummy_bad)
            d_loss = 0.5*d_losst1 + 0.5*d_losst2 + 0.5*d_lossf1 + 0.5*d_lossf2
#             d_loss1 = D.train_on_batch(d_in_true, dummy_good)
#             d_loss2 = D.train_on_batch(d_in_fake, dummy_bad)
#             d_loss = d_loss1 + d_loss2
            # ----------------------- #

            # ----- G training ----- #
            # Latent space sampling
            W1 = np.random.normal(0.0, 1.0, size = [train_size, latent_size])
            W2 = np.random.normal(0.0, 1.0, size = [train_size, latent_size])
            # soft labels
            dummy_good = np.ones(train_size)*0.9 + np.random.uniform(-0.02, 0.02, train_size)
            # train on batch
            gan_in = [W1, W2, X[..., input_flag]]
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
        record_temp_trans = model_mix.evaluate_generator(gen_valid1, verbose=1)
        record_temp = model_mix.evaluate_generator(gen_valid2, verbose=1)
        # Backup validation loss
        V_LOSS[i] = record_temp
        V_LOSS_TRANS[i] = record_temp_trans
        # Overwrite loss info
        LOSS = {'GAN_LOSS':GAN_LOSS, 'D_LOSS':D_LOSS, 
                'V_LOSS':V_LOSS, 'V_LOSS_TRANS':V_LOSS_TRANS}
        np.save(hist_path, LOSS)

        #record_temp += record_temp_trans 
        record_temp = record_temp_trans

        if record - record_temp > min_del:
            print('Validation loss improved from {} to {}'.format(record, record_temp))
            record = record_temp
            tol = 0
            print('tol: {}'.format(tol))
            # save
            print('save to: {}\n\t{}'.format(G_path, D_path))
            model_mix.save(G_path)
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
