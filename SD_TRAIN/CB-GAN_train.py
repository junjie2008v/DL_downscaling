# general tools
import sys
from glob import glob

# data tools
import time
import numpy as np
from random import shuffle
from scipy.ndimage import gaussian_filter

# deep learning tools
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

#
from scipy.ndimage import gaussian_filter

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/')
from namelist import *
import data_utils as du
import model_utils as mu
import train_utils as tu


# class temp_gen(keras.utils.Sequence):
#     def __init__(self, filename, labels, input_flag, output_flag):
#         '''
#         Data generator class for Keras models
#             - filename: list of numpy files, 1 file = 1 batch, 1 list = 1 epoch
#             - labels: e.g., ['input_label', 'target_label']
#             - input_flag: flag of channels, channel last. e.g., [True, False, True]
#             - output_flag: _
#         '''
#         self.filename = filename
#         self.labels = labels
#         self.input_flag = input_flag
#         self.output_flag = output_flag
#         self.filenum = len(self.filename)
#     def __len__(self):
#         return self.filenum
    
#     def __getitem__(self, index):
#         temp_name = self.filename[index]
#         return self.__readfile__(temp_name, self.labels, self.input_flag, self.output_flag)
    
#     def __readfile__(self, temp_name, labels, input_flag, output_flag):
        
#         data_temp = np.load(temp_name, allow_pickle=True)
#         X = data_temp[()][labels[0]][..., input_flag] # channel last
#         Y = data_temp[()][labels[1]][..., output_flag]
        
#         X[..., 0] = gaussian_filter(X[..., 0], 4)
#         return [X], [Y]

def cycle_graph_clean(G1, D1, G2, input_shape):
    '''
    G1 --> D1 --> G2 --> loss
    '''

    ELEV_IN = keras.layers.Input(shape=(None, None, 1))
        
    G1.trainable = True
    G2.trainable = False
    D1.trainable = False
    
    # d loss branch
    # ERA LR --> G1 --> D1 --> (calculate D loss) adversarial loss
    GAN_IN = keras.layers.Input(shape=input_shape)
    G1_OUT = G1(GAN_IN)
    D1_OUT = D1(G1_OUT)
        
    # identity loss branch
    # clean LR --> G1 --> (compare with itself) identity loss BP
    ID_IN = keras.layers.Input(shape=input_shape)
    ID_OUT = G1(ID_IN)
    
    # forward cycle
    # ERA LR --> G1 --> G2 --> (compare with itself) cycle consistency loss
    G_OUT_forward = G2(G1_OUT)

    # backward cycle
    # clean LR --> G2 --> G1 --> (compare with itself) cycle consistency loss
    ID_IN_sub = keras.layers.Lambda(lambda x: x[:, :, :, 0][..., None])(ID_IN) # [..., None] for 4-dim tensor
    IN_OUT2 = G2(ID_IN_sub)
    IN_OUT2_concat = keras.layers.Concatenate()([IN_OUT2, ELEV_IN])
    G_OUT_backward = G1(IN_OUT2_concat)

    # define model graph
    cycle_GAN = keras.models.Model([GAN_IN, ID_IN, ELEV_IN], [D1_OUT, ID_OUT, G_OUT_forward, G_OUT_backward]) # G1_OUT, 

    # define optimization algorithm configuration
    opt = keras.optimizers.Adam(lr=1e-4)
    # compile model with weighting of least squares loss and L1 loss
    cycle_GAN.compile(loss=[keras.losses.mean_squared_error,
                            keras.losses.mean_squared_error,
                            keras.losses.mean_squared_error,
                            keras.losses.mean_squared_error,], 
                  loss_weights=[1, 1, 1, 1], 
                  optimizer=opt)
    return cycle_GAN

def cycle_graph_reanalysis(G1, D1, G2, input_shape, elev_in=False):
    '''
    G1 --> D1 --> G2 --> loss
    '''

    ELEV_IN = keras.layers.Input(shape=(None, None, 1))
        
    G1.trainable = True
    G2.trainable = False
    D1.trainable = False
    
    # d loss branch
    # ERA LR --> G1 --> D1 --> (calculate D loss) adversarial loss
    GAN_IN = keras.layers.Input(shape=input_shape)
    G1_OUT = G1(GAN_IN)
    D1_OUT = D1(G1_OUT)
        
    # identity loss branch
    # clean LR --> G1 --> (compare with itself) identity loss BP
    ID_IN = keras.layers.Input(shape=input_shape)
    ID_OUT = G1(ID_IN)
    
    # forward cycle
    # ERA LR --> G1 --> G2 --> (compare with itself) cycle consistency loss
    G1_OUT_concat = keras.layers.Concatenate()([G1_OUT, ELEV_IN])
    G_OUT_forward = G2(G1_OUT_concat)
    
    # backward cycle
    # clean LR --> G2 --> G1 --> (compare with itself) cycle consistency loss
    IN_concat = keras.layers.Concatenate()([ID_IN, ELEV_IN])
    IN_OUT2 = G2(IN_concat)
    G_OUT_backward = G1(IN_OUT2)
    
    # define model graph
    cycle_GAN = keras.models.Model([GAN_IN, ID_IN, ELEV_IN], [D1_OUT, ID_OUT, G_OUT_forward, G_OUT_backward])
    
    # define optimization algorithm configuration
    opt = keras.optimizers.Adam(lr=1e-4)
    # compile model with weighting of least squares loss and L1 loss
    cycle_GAN.compile(loss=[keras.losses.mean_squared_error,
                            keras.losses.mean_squared_error,
                            keras.losses.mean_squared_error,
                            keras.losses.mean_squared_error], 
                  loss_weights=[1, 1, 1, 1], 
                  optimizer=opt)
    return cycle_GAN

# A: era/jra/fnl LR
# B: LR PRISM
opt_G = keras.optimizers.Adam(lr=1e-4)
G_A2B = mu.EDSR(96, (None, None, 2), num=8, activation='leaky')
G_A2B.compile(loss=keras.losses.mean_squared_error, optimizer=opt_G)

G_B2A = mu.EDSR(96, (None, None, 1), num=4, activation='leaky')
G_B2A.compile(loss=keras.losses.mean_squared_error, optimizer=opt_G)

opt_D = keras.optimizers.Adam(lr=1e-4)
D_A = mu.flat_descriminator(64, (None, None, 1), activation='leaky')
D_A.compile(loss=keras.losses.mean_squared_error, optimizer=opt_D)
    
D_B = mu.flat_descriminator(64, (None, None, 1), activation='leaky')
D_B.compile(loss=keras.losses.mean_squared_error, optimizer=opt_D)

cycle_GAN_AtoB = cycle_graph_clean(G_A2B, D_B, G_B2A, (None, None, 2))
cycle_GAN_BtoA = cycle_graph_reanalysis(G_B2A, D_A, G_A2B, (None, None, 1))

file_path = BATCH_dir
# paired samples (LR PRISM training domain)
train_paired_64 = glob(file_path+'TMEAN_BATCH_64_TORI_*.npy')
train_paired_96 = glob(file_path+'TMEAN_BATCH_96_TORI_*.npy')

# unpaired samples (blured)
train_unpaired_64 = glob(file_path+'TMEAN_BATCH_64_TORI_*.npy')
train_unpaired_96 = glob(file_path+'TMEAN_BATCH_96_TORI_*.npy')

input_flag = [False, True, False, False, False, True]
output_flag = [False, True, False, False, False, False]

# data generators for valid set
labels = ['batch', 'batch']
#gen_valid = temp_gen(valid_paired_64+valid_paired_96, labels, input_flag_paired, output_flag_paired)

epochs = 100
steps_per = 200
steps_half = 100
train_size = 64
train_half = 32
L_train = steps_per+steps_per
min_del = -999
max_tol = 20 # early stopping with patience

key = 'CB-GAN'
model_name = '{}_TMEAN_LR'.format(key)
model_path = temp_dir+model_name+'.hdf'
hist_path = temp_dir+'LOSS_{}_TMEAN_LR.npy'.format(key)

tol = 0
record = 999
GAN_LOSS_A = np.zeros([int(epochs*L_train), 5])*np.nan
GAN_LOSS_B = np.zeros([int(epochs*L_train), 5])*np.nan
D_LOSS_A = np.zeros([int(epochs*L_train)])*np.nan
D_LOSS_B = np.zeros([int(epochs*L_train)])*np.nan
V_LOSS = np.zeros([epochs])*np.nan

for i in range(epochs):
    print('epoch = {}'.format(i))
    start_time = time.time()
    # ----- shuffle ----- #
    shuffle(train_paired_64)
    shuffle(train_paired_96)
    shuffle(train_unpaired_64)
    shuffle(train_unpaired_96)
    # ------------------- #
    
    temp_paired = train_paired_64[:steps_per] + train_paired_96[:steps_per]
    temp_unpaired = train_unpaired_64[:steps_per] + train_unpaired_96[:steps_per]
    
    for j in range(L_train):
        
        # ----- File pipeline ----- #
        # (samples, size, size, channels), channels = (LR TMEAN, LR ELEV)
        inds = du.shuffle_ind(batch_size)[:train_size]
        X_paired = np.load(temp_paired[j], allow_pickle=True)[()]['batch'][inds, ...]
        X_unpaired = np.load(temp_unpaired[j], allow_pickle=True)[()]['batch'][inds, ...]
        
        IN_paired = X_paired[..., input_flag]
        IN_paired_sub = IN_paired[..., 0][..., None]
        IN_paired_elev = IN_paired[..., 1][..., None]
        OUT_paired = X_paired[..., output_flag]
        
        IN_unpaired = X_unpaired[..., input_flag]
        # ----- artifical blur ----- #
        filter_std = 8 + np.random.uniform(low=-2, high=2, size=(train_size))
        for k in range(train_size):
            zero_mask = IN_unpaired[k, ..., 0]==0
            temp_blur = gaussian_filter(IN_unpaired[k, ..., 0], filter_std[k])
            temp_blur[zero_mask] = np.nan
            IN_unpaired[k, ..., 0] = du.fillzero(du.norm_std(temp_blur))
        # -------------------------- #
        IN_unpaired_sub = IN_unpaired[..., 0][..., None]
        IN_unpaired_elev = IN_unpaired[..., 1][..., None]
        
        G_A2B.train_on_batch([IN_unpaired], [-1*IN_unpaired_elev]);
    
        # ----- train AtoB generators ----- #
        dummy_good = np.ones(train_size)*0.9 + np.random.uniform(-0.02, 0.02, train_size)
        # the order of loss: d loss, identity loss, cycle loss
        gan_loss_b = cycle_GAN_AtoB.train_on_batch([IN_unpaired, IN_paired, IN_unpaired_elev], 
                                                   [dummy_good, IN_paired_sub, IN_unpaired_sub, IN_paired_sub])        
        # ----- train D_B ----- #
        # soft labels
        dummy_bad = np.ones(train_size)*0.1 + np.random.uniform(-0.02, 0.02, train_size)
        dummy_good = np.ones(train_size)*0.9 + np.random.uniform(-0.02, 0.02, train_size)
        
        fake_B_paired = G_A2B.predict([IN_paired])
        fake_B_unpaired = G_A2B.predict([IN_unpaired])
        real_B = OUT_paired[:train_size, ...]
        d_b_loss1 = D_B.train_on_batch(fake_B_paired[:train_half], dummy_bad[:train_half])
        d_b_loss1 += D_B.train_on_batch(fake_B_unpaired[:train_half], dummy_bad[:train_half])
        d_b_loss2 = D_B.train_on_batch(real_B, dummy_good)
        
        # ----- train BtoA generators ----- #
        dummy_good = np.ones(train_size)*0.9 + np.random.uniform(-0.02, 0.02, train_size)
        gan_loss_a = cycle_GAN_BtoA.train_on_batch([IN_paired_sub, IN_unpaired_sub, IN_paired_elev],
                                                   [dummy_good, IN_unpaired_sub, IN_paired_sub, IN_unpaired_sub])
        
        # ----- train D_A ----- #
        fake_A = G_B2A.predict([IN_paired_sub])[:train_size, ...]
        real_A = IN_unpaired_sub[:train_size, ...]
        d_a_loss1 = D_A.train_on_batch(fake_A, dummy_bad)
        d_a_loss2 = D_A.train_on_batch(real_A, dummy_good)
        
        # ----- Backup training loss ----- #
        D_LOSS_B[i*L_train+j] = 0.5*(d_b_loss1 + d_b_loss2)
        D_LOSS_A[i*L_train+j] = 0.5*(d_a_loss1 + d_a_loss2)
        GAN_LOSS_B[i*L_train+j, :] = gan_loss_b
        GAN_LOSS_A[i*L_train+j, :] = gan_loss_a
        # -------------------------------- #
        if j%50 == 0:
            print('\t{} step loss = {}'.format(j, gan_loss_b))
            
    record_temp = np.sum(gan_loss_b[2:]) 
    #= G_A2B.evaluate_generator(gen_valid, verbose=1)
    # Backup validation loss
    V_LOSS[i] = record_temp
    # Overwrite loss info
    LOSS = {'GAN_LOSS_A':GAN_LOSS_A,
            'GAN_LOSS_B':GAN_LOSS_B,
            'D_LOSS_A': D_LOSS_A,
            'D_LOSS_B': D_LOSS_B,
            'V_LOSS':V_LOSS}
    np.save(hist_path, LOSS)

    if record - record_temp > min_del:
        print('Validation loss improved from {} to {}'.format(record, record_temp))
        record = record_temp
        tol = 0
        print('tol: {}'.format(tol))
        # save
        print('save to: {}'.format(model_path))
        G_A2B.save(model_path)
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
    
