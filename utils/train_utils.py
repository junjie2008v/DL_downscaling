# general tools
import sys
import glob

# data tools
import random
import numpy as np

# deep learning tools
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
from data_utils import *

def dummy_loader(model_path):
    print('Import model:\n{}'.format(model_path))
    backbone = keras.models.load_model(model_path, compile=False)
    W = backbone.get_weights()
    return W

def parser_handler(args):
    '''
    Convert argparse positional dictionary into variables
    Specified for downscaling model training
    '''
    args['c1'] = int(args['c1'])
    args['c2'] = int(args['c2'])
    
    # arguments handling
    if args['v'] in ['TMAX', 'TMIN', 'TMEAN', 'PCT']:
        VAR = args['v']
    else:
        raise ValueError('Wrong variable')

    if args['s'] == 'annual':
        seasons = ['djf', 'mam', 'jja', 'son']
    elif args['s'] == 'summer':
        seasons = ['jja', 'son']
    elif args['s'] == 'winter':
        seasons = ['djf', 'mam']
    elif args['s'] in ['djf', 'mam', 'jja', 'son']:
        seasons = [args['s']]
    else:
        raise ValueError('Wrong season')

    if args['c1'] == 1:
        input_flag = [False, True, False, False, False, False]
    elif args['c1'] == 2:
        input_flag = [False, True, False, False, True, False]
    elif args['c1'] == 3:
        input_flag = [False, True, False, False, True, True]
    elif args['c1'] == 4:
        input_flag = [False, True, True, False, True, True] 
    elif args['c1'] == 5:
        input_flag = [False, True, True, True, True, True]
    else:
        raise ValueError('Wrong input channel numbers')
    
    if args['c2'] == 1:
        output_flag = [True, False, False, False, False, False]
    elif args['c2'] == 2:
        output_flag = [True, False, False, False, True, False]
    elif args['c2'] == 3:
        output_flag = [True, True, False, False, True, False]
    else:
        raise ValueError('Wrong target numbers')
    
    return VAR, seasons, input_flag, output_flag

class grid_grid_gen(keras.utils.Sequence):
    def __init__(self, filename, labels, input_flag, output_flag, sign_flag=False):
        '''
        Data generator class for Keras models
            - filename: list of numpy files, 1 file = 1 batch, 1 list = 1 epoch
            - labels: e.g., ['input_label', 'target_label']
            - input_flag: flag of channels, channel last. e.g., [True, False, True]
            - output_flag: _
        '''
        self.filename = filename
        self.labels = labels
        self.input_flag = input_flag
        self.output_flag = output_flag
        self.sign_flag = sign_flag
        
        self.filenum = len(self.filename)
    def __len__(self):
        return self.filenum
    
    def __getitem__(self, index):
        temp_name = self.filename[index]
        return self.__readfile__(temp_name, self.labels, self.input_flag, self.output_flag, self.sign_flag)
    
    def __readfile__(self, temp_name, labels, input_flag, output_flag, sign_flag):
        
        data_temp = np.load(temp_name, allow_pickle=True)
        X = data_temp[()][labels[0]][..., input_flag] # channel last
        Y = data_temp[()][labels[1]][..., output_flag]
        if sign_flag:
            return [X], [-1*Y]
        else:
            return [X], [Y]
    
class grid_grid_gen_multi(keras.utils.Sequence):
    def __init__(self, filename, labels, input_flag, output_flag):
        '''
        Data generator class for Keras models (multiple outputs)
            - split from grid_grid_gen() for speeding up. 
        '''
        self.filename = filename
        self.labels = labels
        self.input_flag = input_flag
        self.output_flag = output_flag

        self.filenum = len(self.filename)
    def __len__(self):
        return self.filenum
    
    def __getitem__(self, index):
        temp_name = self.filename[index]
        return self.__readfile__(temp_name, self.labels, self.input_flag, self.output_flag)
    
    def __readfile__(self, temp_name, labels, input_flag, output_flag):
        
        data_temp = np.load(temp_name, allow_pickle=True)
        X = data_temp[()][labels[0]][..., input_flag] # channel last
        Y = data_temp[()][labels[1]][..., output_flag]
        return [X], [Y[..., 0][..., None], -1*Y[..., 1:]]

    
class grid_grid_gen_sub(keras.utils.Sequence):
    def __init__(self, filename, labels, input_flag, output_flag, batch_size=200):
        '''
        Data generator class with Gaussian zero-mean noise on inputs.
            - split from grid_grid_gen() for speeding up.
        '''
        self.filename = filename
        self.labels = labels
        self.input_flag = input_flag
        self.output_flag = output_flag
        self.filenum = len(self.filename)
        self.batch_size = 200
    def __len__(self):
        return self.filenum
    
    def __getitem__(self, index):
        temp_name = self.filename[index]
        return self.__readfile__(temp_name, self.labels, self.input_flag, self.output_flag, self.batch_size)
    
    def __readfile__(self, temp_name, labels, input_flag, output_flag, batch_size):
        
        data_temp = np.load(temp_name, allow_pickle=True)
        X = data_temp[()][labels[0]][..., input_flag] # channel last
        Y = data_temp[()][labels[1]][..., output_flag]
        
        inds = shuffle_ind(batch_size)
        X = X[inds, ...][:100]
        Y = Y[inds, ...][:100]
        return [X], [Y]

class grid_grid_gen_noise(keras.utils.Sequence):
    def __init__(self, filename, labels, input_flag, output_flag, latent_size, sampling=1, sign_flag=False):

        self.filename = filename
        self.labels = labels
        self.input_flag = input_flag
        self.output_flag = output_flag
        self.latent_size = latent_size
        self.sampling = sampling
        self.sign_flag = sign_flag
        self.filenum = len(self.filename)
    def __len__(self):
        return self.filenum
    
    def __getitem__(self, index):
        temp_name = self.filename[index]
        return self.__readfile__(temp_name, self.labels, self.input_flag, self.output_flag, self.latent_size, self.sampling, self.sign_flag)
    
    def __readfile__(self, temp_name, labels, input_flag, output_flag, latent_size, sampling, sign_flag):
        
        data_temp = np.load(temp_name, allow_pickle=True)
        X = data_temp[()][labels[0]][..., input_flag] # channel last
        Y = data_temp[()][labels[1]][..., output_flag]
        train_size = len(Y)
        noise = []
        for i in range(sampling):
            noise += [np.random.normal(0.0, 1.0, size = [train_size, latent_size])]
        if sign_flag:
            return noise+[X], [-1*Y]
        else:
            return noise+[X], [Y]

    
def saliency_maps(temp_model, grid_train_temp, range_top, range_ex, batch_size, layer_id=[-1, -2]):
    # allocation
    top_examples = np.empty((range_top[1]-range_top[0], range_ex[1]-range_ex[0]), dtype=int) # indices of (sorted neurons, sorted samples)
    # array that contains gradients, size: top neuron, top exp, size of gradients on the input end
    top_gradients = np.empty((range_top[1]-range_top[0], range_ex[1]-range_ex[0],)+grid_train_temp.shape[1:]) 
    batch_i = list(range(0, grid_train_temp.shape[0], batch_size)) + [grid_train_temp.shape[0]] # batch samples
    # get weight from the output end
    weights = temp_model.layers[layer_id[0]].get_weights()[0].ravel()
    top_neurons = weights.argsort()[::-1][range_top[0]:range_top[1]] # most activated neurals
    # loop over neurons
    print('Sorted order | neuron index | neuron weights')
    for n, neuron in enumerate(top_neurons):
        print(' {} |  {}  | {}'.format(n, neuron, weights[neuron])) # order, index of neuron, weights
        # define the activation of neurons as a backend function (for sorting the top examples)
        act_func = K.function([temp_model.input, K.learning_phase()], [temp_model.layers[layer_id[1]].output[:, neuron]])
        # loss = a monotonic function that takes neurons' final output 
        loss = (temp_model.layers[layer_id[1]].output[:, neuron]-4)**2
        # calculate gradients from loss (output end) to input end
        grads = K.gradients(loss, temp_model.input)[0]
        # standardizing gradients
        grads /= K.maximum(K.std(grads), K.epsilon())
        # define gradients calculation as a backend function
        grad_func = K.function([temp_model.input, K.learning_phase()], [grads])
        # allocation activation array
        act_values = np.zeros(grid_train_temp.shape[0])
        # loop over samples by batch
        for b in range(len(batch_i)-1):
            act_values[batch_i[b]:batch_i[b+1]] = act_func([grid_train_temp[batch_i[b]:batch_i[b+1]], 0])[0]
        # sort activation values and reteave examples index / gradients
        top_examples[n] = act_values.argsort()[::-1][range_ex[0]:range_ex[1]]
        top_gradients[n, ...] = -grad_func([grid_train_temp[top_examples[n]], 0])[0]  
    return top_neurons, top_examples, top_gradients

