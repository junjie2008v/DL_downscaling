# general tools
import sys
import time
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

# macros
N_input = 1
N_output = 1
l = [5e-5, 5e-5] # G lr; D lr
epochs = 150
lmd = 1e-3
activation='relu' # leakyReLU instead of ReLU
pool=False # stride convolution instead of maxpooling
latent_size = 96
mapping_size = 96
N = [48, 96, 192, 384]
key = 'HALF'
# training settings
tol = 0
min_del = 0
max_tol = 10 # early stopping with patience
input_flag = [False, True]
output_flag = [True, False]
inout_flag = [True, True]
labels = ['batch', 'batch']
batch_size = 200
train_size = 100
record = 999
# ----- G ----- #
input_size = (None, None, N_input)
G_style = mu.UNET_STYLE_half(N, input_size, latent_size, mapping_size, pool=pool, activation=activation)
opt_G = keras.optimizers.Adam(lr=0) # <--- compile G for validation only
print('Compiling G')
G_style.compile(loss=keras.losses.mean_absolute_error, optimizer=opt_G)

# ----- D ----- #
input_size = (None, None, N_input+1)
D = mu.vgg_descriminator(N, input_size)
opt_D = keras.optimizers.Adam(lr=l[1])
print('Compiling D')
D.compile(loss=keras.losses.mean_squared_error, optimizer=opt_D)
D.trainable = False
for layer in D.layers:
    layer.trainable = False

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
file_path = BATCH_dir + 'temp_batches/'
trainfiles = glob(file_path+'ETOPO*64*.npy')+glob(file_path+'ETOPO*96*.npy')+glob(file_path+'ETOPO*128*.npy')
validfiles = glob(file_path+'ETOPO*80*.npy')
# shuffle filenames
shuffle(trainfiles)
shuffle(validfiles)
#
L_train = len(trainfiles)
gen_valid = tu.grid_grid_gen_noise(validfiles, labels, input_flag, output_flag, latent_size, sampling=1)

# model names
G_name = '{}_G_STYLE'.format(key)
D_name = '{}_D_STYLE'.format(key)
G_path = temp_dir+G_name+'.hdf'
D_path = temp_dir+D_name+'.hdf'
hist_path = temp_dir+'{}_LOSS_STYLE.npy'.format(key)

# loss backup
GAN_LOSS = np.zeros([int(epochs*L_train), 3])*np.nan
D_LOSS = np.zeros([int(epochs*L_train)])*np.nan
V_LOSS = np.zeros([epochs])*np.nan

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
        Wf2 = np.random.normal(0.0, 1.0, size = [train_size, latent_size])
        # soft labels
        dummy_bad = np.ones(train_size)*0.1 + np.random.uniform(-0.02, 0.02, train_size)
        dummy_good = np.ones(train_size)*0.9 + np.random.uniform(-0.02, 0.02, train_size)
        # get G_output (channel last)
        g_in = [Wf2, X[..., input_flag]]
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
        Wf2 = np.random.normal(0.0, 1.0, size = [train_size, latent_size])
        # soft labels
        dummy_good = np.ones(train_size)*0.9 + np.random.uniform(-0.02, 0.02, train_size)
        # train on batch
        gan_in = [Wf2, X[..., input_flag]]
        gan_target = [X[..., output_flag], dummy_good]
        gan_loss = GAN.train_on_batch(gan_in, gan_target)
        # ---------------------- #
        
        # ----- Backup training loss ----- #
        D_LOSS[i*L_train+j] = d_loss
        GAN_LOSS[i*L_train+j, :] = gan_loss
        # -------------------------------- #
        if j%40 == 0:
            print('\t{} step loss = {}'.format(j, gan_loss))

    # on epoch-end
    record_temp = G_style.evaluate_generator(gen_valid, verbose=1)
    
    # Backup validation loss
    V_LOSS[i] = record_temp
    # Overwrite loss info
    LOSS = {'GAN_LOSS':GAN_LOSS, 'D_LOSS':D_LOSS, 'V_LOSS':V_LOSS}
    np.save(hist_path, LOSS)

    # early stopping/check point callbacks
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