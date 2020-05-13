import glob
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
#from keras_contrib.layers.normalization import InstanceNormalization

def stride_conv(X, channel, pool_size, activation='relu', name='X'):
    X = keras.layers.Conv2D(channel, pool_size, strides=(pool_size, pool_size), padding='valid', 
                            use_bias=False, kernel_initializer='he_normal', 
                            name=name+'_stride_conv')(X)
    X = keras.layers.BatchNormalization(axis=3, name=name+'_stride_conv_bn')(X)
    if activation == 'relu':
        # ReLU
        X = keras.layers.ReLU(name=name+'_stride_conv_relu')(X)
    elif activation == 'leaky':
        # LeakyReLU
        X = keras.layers.LeakyReLU(alpha=0.3, name=name+'_stride_conv_leaky')(X)
    return X

def DENSE_stack(X, units):
    L = len(units)
    for i in range(L):
        X = keras.layers.Dense(units[i], use_bias=False, kernel_initializer='he_normal')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.ReLU()(X)
    return X

def CONV_stack(X, channel, kernel_size, stack_num, activation='relu', name='conv_stack'):
    '''
    Stacked convolution-BN-ReLU blocks
    '''
    for i in range(stack_num):
        X = keras.layers.Conv2D(channel, kernel_size, padding='same', use_bias=False, 
                                kernel_initializer='he_normal', name=name+'_stack{}_conv'.format(i))(X)
        X = keras.layers.BatchNormalization(axis=3, name=name+'_stack{}_bn'.format(i))(X)
        if activation == 'relu':
            # ReLU
            X = keras.layers.ReLU(name=name+'_stack{}_relu'.format(i))(X)
        elif activation == 'leaky':
            # LeakyReLU
            X = keras.layers.LeakyReLU(alpha=0.3, name=name+'_stack{}_leaky'.format(i))(X)
    return X

# UNet
def UNET_left(X, channel, kernel_size=3, pool_size=2, pool=True, activation='relu', name='left0'):
    '''
    U-Net encoder block.
    '''
    if pool:
        X = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), name='{}_pool'.format(name))(X)
    else:
        X = stride_conv(X, channel, pool_size, activation=activation, name=name)
    X = CONV_stack(X, channel, kernel_size, stack_num=1, activation=activation, name=name)
    return X

def UNET_right(X, X_left, channel, kernel_size=3, pool_size=2, activation='relu', name='right0'):
    '''
    U-Net decoder block with transpose conv.
    '''
    X = keras.layers.Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size), 
                                     padding='same', name=name+'_trans_conv')(X)
    X = CONV_stack(X, channel, kernel_size, stack_num=1, activation=activation, name=name+'_conv_after_trans') 
    H = keras.layers.concatenate([X_left, X], axis=3)
    H = CONV_stack(H, channel, kernel_size, stack_num=1, activation=activation, name=name+'_conv_after_concat')
    return H

def UNET_left_style(X, STY, channel, kernel_size=3, pool_size=2, pool=True, activation='relu', noise=False, name='left0'):
    # downsampling layer
    if pool:
        X = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), name='{}_pool'.format(name))(X)
    else:
        X = stride_conv(X, channel, pool_size, activation=activation, name=name)
    # Conv layer
    X = keras.layers.Conv2D(channel, kernel_size, padding='same', use_bias=False, 
                            kernel_initializer='he_normal', name='{}_conv'.format(name))(X)
    # ----- AdaIN ----- #
    # affine transform
    b = keras.layers.Dense(channel, activation=keras.activations.linear, 
                           kernel_initializer='he_normal', name='{}_AdaIN_b'.format(name))(STY)
    b = keras.layers.Reshape([1, 1, channel])(b)
    
    g = keras.layers.Dense(channel, activation=keras.activations.linear, 
                           kernel_initializer='he_normal', name='{}_AdaIN_g'.format(name))(STY)
    g = keras.layers.Reshape([1, 1, channel])(g)  
    # AdaIN
    X = AdaIN()([X, b, g])
    # ----------------- #
    if noise:
        X = keras.layers.GaussianDropout(noise, name=name+'_gaussian_drop')(X)
    if activation == 'relu':
        X = keras.layers.ReLU(name=name+'_relu0')(X)
    elif activation == 'leaky':
        X = keras.layers.LeakyReLU(alpha=0.3, name=name+'_leaky0')(X)
    return X

def UNET_right_style(X, X_left, STY, channel, kernel_size=3, pool_size=2, activation='relu', noise=False, name='right0'):
    # up-sampling
    X = keras.layers.Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size), padding='same',
                                     name=name+'_trans_conv')(X)
    # conv
    X = keras.layers.Conv2D(channel, kernel_size, padding='same', use_bias=False, 
                            kernel_initializer='he_normal', name=name+'_conv0')(X)
    # ----- AdaIN ----- #
    # affine transform
    b = keras.layers.Dense(channel, activation=keras.activations.linear, kernel_initializer='he_normal', 
                           name='{}_AdaIN0_b'.format(name))(STY)
    b = keras.layers.Reshape([1, 1, channel])(b)
    g = keras.layers.Dense(channel, activation=keras.activations.linear, kernel_initializer='he_normal',
                           name='{}_AdaIN0_g'.format(name))(STY)
    g = keras.layers.Reshape([1, 1, channel])(g)
    # AdaIN
    X = AdaIN()([X, b, g])
    # ----------------- #
    if noise:
        X = keras.layers.GaussianDropout(noise, name=name+'_gaussian_drop0')(X)
    if activation == 'relu':
        X = keras.layers.ReLU(name=name+'_relu0')(X)
    elif activation == 'leaky':
        X = keras.layers.LeakyReLU(alpha=0.3, name=name+'_leaky0')(X)
    
    H = keras.layers.concatenate([X_left, X], axis=3)
    # conv
    H = keras.layers.Conv2D(channel, kernel_size, padding='same', use_bias=False, kernel_initializer='he_normal', 
                            name=name+'_conv1')(H)
    # ----- AdaIN ----- #
    # affine transform
    b = keras.layers.Dense(channel, activation=keras.activations.linear, kernel_initializer='he_normal',
                           name='{}_AdaIN1_b'.format(name))(STY)
    b = keras.layers.Reshape([1, 1, channel])(b)
    g = keras.layers.Dense(channel, activation=keras.activations.linear, kernel_initializer='he_normal',
                           name='{}_AdaIN1_g'.format(name))(STY)
    g = keras.layers.Reshape([1, 1, channel])(g)
    # AdaIN
    H = AdaIN()([H, b, g])
    # ----------------- #
    if noise:
        H = keras.layers.GaussianDropout(noise, name=name+'_gaussian_drop1')(H)
    if activation == 'relu':
        H = keras.layers.ReLU(name=name+'_relu1')(H)
    elif activation == 'leaky':
        H = keras.layers.LeakyReLU(alpha=0.3, name=name+'_leaky1')(H)
    return H

def UNET_right_style_2(X, X_left, STY1, STY2, channel, kernel_size=3, pool_size=2, activation='relu', noise=False, name='right0'):
    # up-sampling
    X = keras.layers.Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size), padding='same',
                                     name=name+'_trans_conv')(X)
    # conv
    X = keras.layers.Conv2D(channel, kernel_size, padding='same', use_bias=False, 
                            kernel_initializer='he_normal', name=name+'_conv0')(X)
    # ----- AdaIN ----- #
    # affine transform
    b = keras.layers.Dense(channel, activation=keras.activations.linear, kernel_initializer='he_normal', 
                           name='{}_AdaIN0_b'.format(name))(STY1)
    b = keras.layers.Reshape([1, 1, channel])(b)
    g = keras.layers.Dense(channel, activation=keras.activations.linear, kernel_initializer='he_normal',
                           name='{}_AdaIN0_g'.format(name))(STY1)
    g = keras.layers.Reshape([1, 1, channel])(g)
    # AdaIN
    X = AdaIN()([X, b, g])
    # ----------------- #
    if noise:
        X = keras.layers.GaussianDropout(noise, name=name+'_gaussian_drop0')(X)
    if activation == 'relu':
        X = keras.layers.ReLU(name=name+'_relu0')(X)
    elif activation == 'leaky':
        X = keras.layers.LeakyReLU(alpha=0.3, name=name+'_leaky0')(X)
    
    H = keras.layers.concatenate([X_left, X], axis=3)
    # conv
    H = keras.layers.Conv2D(channel, kernel_size, padding='same', use_bias=False, kernel_initializer='he_normal', 
                            name=name+'_conv1')(H)
    # ----- AdaIN ----- #
    # affine transform
    b = keras.layers.Dense(channel, activation=keras.activations.linear, kernel_initializer='he_normal',
                           name='{}_AdaIN1_b'.format(name))(STY2)
    b = keras.layers.Reshape([1, 1, channel])(b)
    g = keras.layers.Dense(channel, activation=keras.activations.linear, kernel_initializer='he_normal',
                           name='{}_AdaIN1_g'.format(name))(STY2)
    g = keras.layers.Reshape([1, 1, channel])(g)
    # AdaIN
    H = AdaIN()([H, b, g])
    # ----------------- #
    if noise:
        H = keras.layers.GaussianDropout(noise, name=name+'_gaussian_drop1')(H)
    if activation == 'relu':
        H = keras.layers.ReLU(name=name+'_relu1')(H)
    elif activation == 'leaky':
        H = keras.layers.LeakyReLU(alpha=0.3, name=name+'_leaky1')(H)
    return H

def UNET_out_style(X, STY, channel=2, kernel_size=3, pool_size=2, pool=True, activation='relu', name='out'):
    # Conv layer
    X = keras.layers.Conv2D(channel, kernel_size, padding='same', use_bias=False, 
                            kernel_initializer='he_normal', name=name+'_conv')(X)
    # additive noise (not applied)
    # ----- AdaIN ----- #
    # affine transform
    b = keras.layers.Dense(channel, activation=keras.activations.linear, kernel_initializer='he_normal',
                           name='{}_AdaIN_b'.format(name))(STY)
    b = keras.layers.Reshape([1, 1, channel])(b)
    g = keras.layers.Dense(channel, activation=keras.activations.linear, kernel_initializer='he_normal',
                           name='{}_AdaIN_g'.format(name))(STY)
    g = keras.layers.Reshape([1, 1, channel])(g)
    # AdaIN
    X = AdaIN()([X, b, g])
    # ----------------- #
    if activation == 'relu':
        X = keras.layers.ReLU(name=name+'_relu0')(X)
    elif activation == 'leaky':
        X = keras.layers.LeakyReLU(alpha=0.3, name=name+'_leaky0')(X)
    return X

def UNET(layer_N, input_size, input_stack_num=2, pool=True, activation='relu'):
    '''
    U-Net with 8x downsampling rate (3 down- and upsampling levels)
    '''
    IN = keras.layers.Input(input_size, name='unet_in')
    # left blocks
    X_en1 = CONV_stack(IN, layer_N[0], kernel_size=3, stack_num=input_stack_num, activation=activation, name='unet_left0')
    X_en2 = UNET_left(X_en1, layer_N[1], pool=pool, activation=activation, name='unet_left1')
    X_en3 = UNET_left(X_en2, layer_N[2], pool=pool, activation=activation, name='unet_left2')
    # bottom
    X4 = UNET_left(X_en3, layer_N[3], pool=pool, activation=activation, name='unet_bottom')
    # right blocks
    X_de3 = UNET_right(X4, X_en3, layer_N[2], activation=activation, name='unet_right2')
    X_de2 = UNET_right(X_de3, X_en2, layer_N[1], activation=activation, name='unet_right1')
    X_de1 = UNET_right(X_de2, X_en1, layer_N[0], activation=activation, name='unet_right0')
    # output
    OUT = CONV_stack(X_de1, 2, kernel_size=3, stack_num=1, activation=activation, name='unet_out')
    OUT = keras.layers.Conv2D(1, 1, activation=keras.activations.linear, padding='same', name='unet_exit')(OUT)
    model = keras.models.Model(inputs=[IN], outputs=[OUT])
    return model

def UNET_STYLE(N, input_size, latent_lev, latent_size, mapping_size, pool=True, activation='relu', noise=[0.2, 0.1]):
    # mapping network #1
    IN2 = keras.layers.Input(shape=[latent_size], name='mapping_in')
    STY2 = keras.layers.Dense(mapping_size, kernel_initializer='he_normal', name='mapping_hidden0')(IN2)
    
    if activation == 'relu':
        STY2 = keras.layers.ReLU(name='mapping_hidden_relu0')(STY2)
    elif activation == 'leaky':
        STY2 = keras.layers.LeakyReLU(alpha=0.3, name='mapping_hidden_leaky0')(STY2)
        
    for i in range(latent_lev):
        STY2 = keras.layers.Dense(mapping_size, kernel_initializer='he_normal', name='mapping_hidden{}'.format(i+1))(STY2)
        if activation == 'relu':
            STY2 = keras.layers.ReLU(name='mapping_hidden_relu{}'.format(i+1))(STY2)
        elif activation == 'leaky':
            STY2 = keras.layers.LeakyReLU(alpha=0.3, name='mapping_hidden_leaky{}'.format(i+1))(STY2)

    # main UNet
    IN3 = keras.layers.Input(input_size, name='unet_in')
    X_en1 = CONV_stack(IN3, N[0], kernel_size=3, stack_num=2, activation=activation, name='unet_left0')
    X_en2 = UNET_left(X_en1, N[1], pool=pool, activation=activation, name='unet_left1')
    X_en3 = UNET_left(X_en2, N[2], pool=pool, activation=activation, name='unet_left2')
    
    X_de4 = UNET_left_style(X_en3, STY2, N[3], pool=pool, activation=activation, noise=noise[0], name='unet_bottom')
    X_de3 = UNET_right_style(X_de4, X_en3, STY2, N[2], activation=activation, noise=noise[1], name='unet_right2')
    X_de2 = UNET_right_style(X_de3, X_en2, STY2, N[1], activation=activation, noise=noise[1], name='unet_right1')
    X_de1 = UNET_right_style(X_de2, X_en1, STY2, N[0], activation=activation, noise=noise[1], name='unet_right0')
    # output
    OUT = UNET_out_style(X_de1, STY2, activation=activation, name='unet_out')
    OUT = keras.layers.Conv2D(1, 1, activation=keras.activations.linear, padding='same', name='unet_exit')(OUT)
    G_style = keras.models.Model(inputs=[IN2, IN3], outputs=[OUT])
    return G_style

def UNET_MAP(N, input_size, latent_lev, latent_size, mapping_size, pool=True, activation='relu', noise=[0.2, 0.1]):
    # mapping network #1
    STY2 = keras.layers.Input(shape=[latent_size], name='mapping_in')
    # main UNet
    IN3 = keras.layers.Input(input_size, name='unet_in')
    X_en1 = CONV_stack(IN3, N[0], kernel_size=3, stack_num=2, activation=activation, name='unet_left0')
    X_en2 = UNET_left(X_en1, N[1], pool=pool, activation=activation, name='unet_left1')
    X_en3 = UNET_left(X_en2, N[2], pool=pool, activation=activation, name='unet_left2')
    
    X_de4 = UNET_left_style(X_en3, STY2, N[3], pool=pool, activation=activation, noise=noise[0], name='unet_bottom')
    X_de3 = UNET_right_style(X_de4, X_en3, STY2, N[2], activation=activation, noise=noise[1], name='unet_right2')
    X_de2 = UNET_right_style(X_de3, X_en2, STY2, N[1], activation=activation, noise=noise[1], name='unet_right1')
    X_de1 = UNET_right_style(X_de2, X_en1, STY2, N[0], activation=activation, noise=noise[1], name='unet_right0')
    # output
    OUT = UNET_out_style(X_de1, STY2, activation=activation, name='unet_out')
    OUT = keras.layers.Conv2D(1, 1, activation=keras.activations.linear, padding='same', name='unet_exit')(OUT)
    G_style = keras.models.Model(inputs=[STY2, IN3], outputs=[OUT])
    return G_style

def latent_mix(STY1, STY2, latent_size, activation, name):
    STY1_expand = keras.layers.Reshape([1, latent_size])(STY1)
    STY2_expand = keras.layers.Reshape([1, latent_size])(STY2)
    STYMIX = keras.layers.Concatenate(axis=1)([STY1_expand, STY2_expand])
    STYMIX = keras.layers.LocallyConnected1D(latent_size, 2, strides=1, padding='valid',
                                             kernel_initializer='he_normal', name=name+'_dense')(STYMIX)
    if activation == 'relu':
        STYMIX = keras.layers.ReLU(name=name+'_relu')(STYMIX)
    elif activation == 'leaky':
        STYMIX = keras.layers.LeakyReLU(alpha=0.3, name=name+'_leaky')(STYMIX)
    return STYMIX

def UNET_MAP_MIX(N, input_size, latent_lev, latent_size, mapping_size, pool=True, activation='relu', noise=[0.2, 0.1]):
    # mapping network #1
    STY1 = keras.layers.Input(shape=[latent_size], name='mapping_in1')
    STY2 = keras.layers.Input(shape=[latent_size], name='mapping_in2')
    
    STYMIX0 = latent_mix(STY1, STY2, latent_size, activation, name='mix0')
    # main UNet
    IN3 = keras.layers.Input(input_size, name='unet_in')
    X_en1 = CONV_stack(IN3, N[0], kernel_size=3, stack_num=2, activation=activation, name='unet_left0')
    X_en2 = UNET_left(X_en1, N[1], pool=pool, activation=activation, name='unet_left1')
    X_en3 = UNET_left(X_en2, N[2], pool=pool, activation=activation, name='unet_left2')
    
    X_de4 = UNET_left_style(X_en3, STYMIX0, N[3], pool=pool, activation=activation, 
                            noise=noise[0], name='unet_bottom')
    
    X_de3 = UNET_right_style(X_de4, X_en3, STY1, N[2], activation=activation, noise=noise[1], name='unet_right2')
    X_de2 = UNET_right_style(X_de3, X_en2, STY1, N[1], activation=activation, noise=noise[1], name='unet_right1')
    X_de1 = UNET_right_style(X_de2, X_en1, STY1, N[0], activation=activation, noise=noise[1], name='unet_right0')
    # output
    OUT = UNET_out_style(X_de1, STY1, activation=activation, name='unet_out')
    OUT = keras.layers.Conv2D(1, 1, activation=keras.activations.linear, padding='same', name='unet_exit')(OUT)
    G_style = keras.models.Model(inputs=[STY1, STY2, IN3], outputs=[OUT])
    return G_style

def STYLE_MAP(latent_lev, latent_size, mapping_size, activation):
    IN2 = keras.layers.Input(shape=[latent_size], name='mapping_in')
    STY2 = keras.layers.Dense(mapping_size, kernel_initializer='he_normal', name='mapping_hidden0')(IN2)
    
    if activation == 'relu':
        STY2 = keras.layers.ReLU(name='mapping_hidden_relu0')(STY2)
    elif activation == 'leaky':
        STY2 = keras.layers.LeakyReLU(alpha=0.3, name='mapping_hidden_leaky0')(STY2)
        
    for i in range(latent_lev):
        STY2 = keras.layers.Dense(mapping_size, kernel_initializer='he_normal', name='mapping_hidden{}'.format(i+1))(STY2)
        if activation == 'relu':
            STY2 = keras.layers.ReLU(name='mapping_hidden_relu{}'.format(i+1))(STY2)
        elif activation == 'leaky':
            STY2 = keras.layers.LeakyReLU(alpha=0.3, name='mapping_hidden_leaky{}'.format(i+1))(STY2)
    model = keras.models.Model(inputs=[IN2], outputs=[STY2])
    return model

def UNET_AE(layer_N, input_size, output_channel_num=1, input_stack_num=2, drop_rate=0.05):
    '''
    UNet-AE with 8x downsampling rate (3 down- and upsampling levels)
    '''
    if drop_rate <= 0:
        drop_flag = False
    else:
        drop_flag = True
    # ---------- model ---------- #
    IN = keras.layers.Input(input_size)
    # left blocks
    X_en1 = CONV_stack(IN, layer_N[0], kernel_size=3, stack_num=input_stack_num)
    X_en2 = UNET_left(X_en1, layer_N[1])
    X_en3 = UNET_left(X_en2, layer_N[2])
    # bottom
    X4 = UNET_left(X_en3, layer_N[3])
    if drop_flag:
        X4 = keras.layers.SpatialDropout2D(drop_rate)(X4)
    # right blocks
    X_de3 = UNET_right(X4, X_en3, layer_N[2])
    if drop_flag:
        X_de3 = keras.layers.SpatialDropout2D(drop_rate)(X_de3)
    X_de2 = UNET_right(X_de3, X_en2, layer_N[1])
    if drop_flag:
        X_de2 = keras.layers.SpatialDropout2D(drop_rate)(X_de2)
    X_de1 = UNET_right(X_de2, X_en1, layer_N[0])
    if drop_flag:
        X_de1 = keras.layers.SpatialDropout2D(drop_rate)(X_de1)
    # output (supervised)
    OUT1 = CONV_stack(X_de1, 2*output_channel_num, kernel_size=3, stack_num=1)
    OUT1 = keras.layers.Conv2D(1, 1, activation=keras.activations.linear, name='HR_temp')(OUT1)
    # output (unsupervised)
    OUT2 = CONV_stack(X_de1, 2*output_channel_num, kernel_size=3, stack_num=1)
    OUT2 = keras.layers.Conv2D(output_channel_num, output_channel_num, padding='same',
                               activation=keras.activations.linear, name='HR_elev')(OUT2)
    model = keras.models.Model(inputs=[IN], outputs=[OUT1, OUT2])
    return model


# UNet++
def XNET_left(X, channel, kernel_size=3):
    # down-sampling
    X_pool = keras.layers.MaxPooling2D(pool_size=(2, 2))(X)
    # conv+BN blocks
    X_conv = CONV_stack(X_pool, channel, kernel_size, stack_num=2)
    return X_conv

def XNET_right(X_conv, X_list, channel, kernel_size=3, pool_size=2):
    # up-sampling + conv + concat
    X_unpool = keras.layers.Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size), padding='same')(X_conv)
    X_unpool = keras.layers.concatenate([X_unpool]+X_list, axis=3)
    # conv+BN blocks 
    X_conv = CONV_stack(X_unpool, channel, kernel_size, stack_num=2)
    return X_conv

def XNET(layer_N, input_size, input_stack_num=2):
    '''
    UNet++ with 8x downsampling rate
    '''
    IN = keras.layers.Input(input_size)
    X11_conv = CONV_stack(IN, layer_N[0], kernel_size=3, stack_num=input_stack_num)
    # down-sampling blocks
    X21_conv = XNET_left(X11_conv, layer_N[1])
    X31_conv = XNET_left(X21_conv, layer_N[2])
    X41_conv = XNET_left(X31_conv, layer_N[3])
    # up-sampling blocks 2
    X12_conv = XNET_right(X21_conv, [X11_conv], layer_N[0]) 
    X22_conv = XNET_right(X31_conv, [X21_conv], layer_N[1])
    X32_conv = XNET_right(X41_conv, [X31_conv], layer_N[2])
    # up-sampling blocks 3
    X13_conv = XNET_right(X22_conv, [X11_conv, X12_conv], layer_N[0]) 
    X23_conv = XNET_right(X32_conv, [X21_conv, X22_conv], layer_N[1]) 
    # up-sampling blocks 4
    X14_conv = XNET_right(X23_conv, [X11_conv, X12_conv, X13_conv], layer_N[0]) 
    # output end
    OUT = keras.layers.Conv2D(1, 1, activation=keras.activations.linear)(X14_conv)
    model = keras.models.Model(inputs=[IN], outputs=[OUT])
    return model

def vgg_descriminator(N, input_size, pool_size=2, activation='leaky'):
    '''
    xxxxxx
    '''
    IN = keras.layers.Input(input_size, name='d_in')

    X = CONV_stack(IN, N[0], kernel_size=3, stack_num=2, activation=activation, name='d_conv1')
    X = stride_conv(X, N[0], pool_size, activation=activation, name='d_down1')

    X = CONV_stack(X, N[1], kernel_size=3, stack_num=2, activation=activation, name='d_conv2')
    X = stride_conv(X, N[1], pool_size, activation=activation, name='d_down2')

    X = CONV_stack(X, N[2], kernel_size=3, stack_num=2, activation=activation, name='d_conv3')

    X = keras.layers.GlobalAveragePooling2D(name='d_down3')(X)
    FLAG = keras.layers.Dense(1, activation=keras.activations.sigmoid)(X) #, 

    return keras.models.Model(inputs=[IN], outputs=[FLAG])

class AdaIN(keras.layers.Layer):
    '''
    AdaIN layer in style-GAN v1
    '''
    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True, **kwargs):
        super(AdaIN, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
    
    def build(self, input_shape):
    
        dim = input_shape[0][self.axis]
#         if dim is None:
#             raise ValueError('Axis ' + str(self.axis) + ' of '
#                              'input tensor should have a defined dimension '
#                              'but the layer received an input with shape ' +
#                              str(input_shape[0]) + '.')
        super(AdaIN, self).build(input_shape) 
    
    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs[0])
        reduction_axes = list(range(0, len(input_shape)))
        
        beta = inputs[1]
        gamma = inputs[2]

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]
        mean = K.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs[0] - mean) / stddev

        return normed * gamma + beta
    
    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(AdaIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
    
        return input_shape[0]
    
    
    
def RES_Block(X_in, channel, kernel_size, activation, name):

    X = keras.layers.Conv2D(channel, kernel_size, padding='same', use_bias=False, 
                            kernel_initializer='he_normal', name=name+'_conv0')(X_in)
    X = keras.layers.BatchNormalization(axis=3, name=name+'_bn0')(X)
    if activation == 'relu':
        X = keras.layers.ReLU(name=name+'_relu0')(X)
    elif activation == 'leaky':
        X = keras.layers.LeakyReLU(name=name+'_leaky0')(X)
        
    X = keras.layers.Conv2D(channel, kernel_size, padding='same', use_bias=False, 
                            kernel_initializer='he_normal', name=name+'_conv1')(X)
    X = keras.layers.Add()([X_in, X])
    return X

# def RES_Block(X_in, channel, kernel_size, activation, name):

#     X = keras.layers.Conv2D(channel, kernel_size, padding='same', use_bias=False, 
#                             kernel_initializer='he_normal', name=name+'_conv0')(X_in)
#     X = keras.layers.BatchNormalization(axis=3, name=name+'_bn0')(X)
#     if activation == 'relu':
#         X = keras.layers.ReLU(name=name+'_relu0')(X)
#     elif activation == 'leaky':
#         X = keras.layers.LeakyReLU(name=name+'_leaky0')(X)
        
#     X = keras.layers.Add()([X_in, X])
    
#     X = keras.layers.Conv2D(channel, kernel_size, padding='same', use_bias=False, 
#                             kernel_initializer='he_normal', name=name+'_conv1')(X)
#     X = keras.layers.BatchNormalization(axis=3, name=name+'_bn1')(X)
#     if activation == 'relu':
#         X = keras.layers.ReLU(name=name+'_relu1')(X)
#     elif activation == 'leaky':
#         X = keras.layers.LeakyReLU(name=name+'_leaky1')(X)
#     return X


def EDSR(channel, input_size, kernel_size=3, num=4, activation='relu'):
    '''
    EDSR (Enhanced Deep Residual Network)
    '''
    IN = keras.layers.Input(shape=input_size)
    X = keras.layers.Conv2D(channel, kernel_size, padding='same', use_bias=False, 
                            kernel_initializer='he_normal', name='edsr_conv0')(IN)
    X = keras.layers.BatchNormalization(axis=3, name='edsr_bn0')(X)
    if activation == 'relu':
        X = keras.layers.ReLU(name='edsr_relu0')(X)
    elif activation == 'leaky':
        X = keras.layers.LeakyReLU(name='edsr_leaky0')(X)
    X_res = X
    for i in range(num):
        X_res = RES_Block(X_res, channel, kernel_size, activation, name='res_block{}'.format(i))
        
    X_res = keras.layers.Conv2D(channel, kernel_size, padding='same', use_bias=False, 
                                kernel_initializer='he_normal', name='edsr_conv1')(X_res)
    X_res = keras.layers.BatchNormalization(axis=3, name='edsr_bn1')(X_res)
    if activation == 'relu':
        X_res = keras.layers.ReLU(name='edsr_relu1')(X_res)
    elif activation == 'leaky':
        X_res = keras.layers.LeakyReLU(name='edsr_leaky1')(X_res)
    X = keras.layers.Add()([X, X_res])

    X = keras.layers.Conv2D(channel, kernel_size, padding='same', use_bias=False, 
                            kernel_initializer='he_normal', name='edsr_conv2')(X)
    X = keras.layers.BatchNormalization(axis=3, name='edsr_bn2')(X)
    if activation == 'relu':
        X = keras.layers.ReLU(name='edsr_relu2')(X)
    elif activation == 'leaky':
        X = keras.layers.LeakyReLU(name='edsr_leaky2')(X)

    OUT = keras.layers.Conv2D(1, 1, activation=keras.activations.linear, padding='same', name='edsr_exit')(X)

    return keras.models.Model([IN], [OUT])

def flat_descriminator(N, input_size, activation='leaky'):
    '''
    CNN with stacked convolutional layers
    '''
    IN = keras.layers.Input(input_size, name='d_in')

    X = CONV_stack(IN, N, kernel_size=3, stack_num=3, activation=activation, name='d_conv1')

    X = keras.layers.GlobalAveragePooling2D(name='d_avepool')(X)
    FLAG = keras.layers.Dense(1, activation=keras.activations.sigmoid)(X) #, 

    return keras.models.Model(inputs=[IN], outputs=[FLAG])

