# -*- coding:utf-8 -*-
import sys
sys.path.append('dcgan_code/')
sys.path.append('dcgan_code/lib')
sys.path.append('dcgan_code/mnist')

import os
import json
from time import time
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib

import theano
import theano.tensor as T
from theano.gpuarray.dnn import dnn_conv

import lasagne.layers as ll
import lasagne.objectives as lo
import lasagne.updates as lu

from lib import activations
from lib import updates
from lib import inits
from lib.vis import grayscale_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data
from lib.metrics import nnc_score, nnd_score

from load import mnist_with_valid_set

import binary_mnist

IS_BINARY = True
k = 1             # # of discrim updates for each gen update
l2 = 2.5e-5       # l2 weight decay
b1 = 0.5          # momentum term of adam
nc = 1            # # of channels in image
NUM_Y = 10           # # of classes
MINIBATCH_SIZE = 128      # # of examples in batch
npx = 28          # # of pixels width/height of images
Z_SIZE = 100          # # of dim for Z
Z_MIN=-1.0
Z_MAX=+1.0

V_MIN=0.0
V_MAX=+1.0

#ngfc = 1024       # # of gen units for fully connected layers
#ndfc = 1024       # # of discrim units for fully connected layers
#ngf = 64          # # of gen filters in first conv layer
#ndf = 64          # # of discrim filters in first conv layer
#nx = npx*npx*nc   # # of dimensions in X
niter = 20       # # of iter at starting learning rate
#niter_decay = 3000 # # of iter to linearly decay learning rate to zero
niter_decay = 200 # # of iter to linearly decay learning rate to zero
#lr = 0.0001/2.0       # initial learning rate for adam
#lr = 0.00001       # initial learning rate for adam
#lr = 0.0001/3.0       # initial learning rate for adam
#lr = 0.001       # initial learning rate for adam
lr      = 0.0001        # initial learning rate for adam
lr2     = 0.000001      # 2nd LR

LR_DECAY_ALGO=1    # 1 or 2

VER='1.2'
VER_DESC='LR decay algorithm %d, BinDeconv On/Off each layer' % (LR_DECAY_ALGO)

def reshapeX_forBNN(X):
    # [-1.0, +1.0]
    return (V_MAX-V_MIN)*(floatX(X).reshape((-1, nc, npx, npx))/255.0) + V_MIN

def reshapeY_forBNN(Y):
    one_hot = floatX(np.eye(10)[Y])
    return (V_MAX-V_MIN)*one_hot + V_MIN

def sample_z(a_batch_size, a_z_size):
    return floatX(np_rng.uniform(Z_MIN, Z_MAX, size=(a_batch_size, a_z_size)))

def printVal(name, V):
    print( '%s:' % name, type(V), 'ndim:', V.ndim, ':', V.shape )

def print_out(out, name):
    #print '%s:\n%s' %(name, out)
    print '%s[0]: shape:%s, mean:%f, std:%f, [%f,%f]' % ( name, out[0].shape, out[0].mean(), out[0].std(), out[0].min(), out[0].max())
    print '%s   : shape:%s, mean:%f, std:%f, [%f,%f]' % ( name, out.shape, out.mean(), out.std(), out.min(), out.max())

def transform(X):
    return (floatX(X)/255.).reshape(-1, nc, npx, npx)

def inverse_transform(X):
    X = X.reshape(-1, npx, npx)
    return X

def expandRows( tensor, obj_num_rows ):
    t_len = len(tensor)
    tmp_row = tensor[t_len-1:].copy()
    for rn in range(t_len, obj_num_rows):
        tensor = np.append( tensor, tmp_row, axis=0 )

    return tensor

def gen_samples(n, nbatch, ysize):
    samples = []
    labels = []
    n_gen = 0
    for i in range(n/nbatch):
        ymb = floatX(OneHot(np_rng.randint(0, 10, nbatch), ysize))
        zmb = sample_z(nbatch, Z_SIZE)
        xmb, ot_lYS, ot_G3_1, ot_G3_2, ot_G10, ot_G11, ot_G12 = _gen(zmb, ymb)
        samples.append(xmb)
        labels.append(np.argmax(ymb, axis=1))
        n_gen += len(xmb)

    # fraction part
    n_left = n-n_gen
    ymb = floatX(OneHot(np_rng.randint(0, 10, nbatch), ysize))
    zmb = sample_z(nbatch, Z_SIZE)
    xmb, ot_lYS, ot_G3_1, ot_G3_2, ot_G10, ot_G11, ot_G12 = _gen(zmb, ymb)

    xmb = xmb[0:n_left]

    samples.append(xmb)
    labels.append(np.argmax(ymb, axis=1))
    return np.concatenate(samples, axis=0), np.concatenate(labels, axis=0)

#
#
# main
#
#
bin_mnist = binary_mnist.BinaryMnist(is_binary=IS_BINARY)

print sys.argv[0], 'ver.', VER, VER_DESC, '/', bin_mnist.getVersionDesc()

# Loading MNIST datasets
#  trX:training set (image 28x28)
#  vaX:varidation set （ trX of 50000<=index after shuffle ）
#  teX:test set
#  trY:training labels
#  vaY:validation labels ( trY of 50000<=index after shuffle )
#  teY:test labels

trX, vaX, teX, trY, vaY, teY = mnist_with_valid_set()

#

trX0 = trX
trY0 = trY

if IS_BINARY:
    trX = reshapeX_forBNN(trX)
    trY = reshapeY_forBNN(trY)
    teX = reshapeX_forBNN(teX)
    teY = reshapeY_forBNN(teY)

print 'trX:max,min', np.max(trX), np.min(trX)
print('#### trX:', trX.shape)
print('#### vaX:', vaX.shape)
print('#### teX:', teX.shape)

print('#### trY:', trY.shape)
print('#### vaY:', vaY.shape)
print('#### teY:', teY.shape)

ntrain, nval, ntest = len(trX), len(vaX), len(teX)

#
# Making Directories
#
desc = 'b_dcgan'
model_dir = 'models/%s'%desc
samples_dir = 'samples/%s'%desc
if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
bce = T.nnet.binary_crossentropy

X = T.tensor4('X')
Z = T.matrix('Z')
Y = T.matrix('Y')

bin_mnist.getProperties()

## Generator(G)
gX_layers, out_lYS, out_G3_1, out_G3_2, out_G10, out_G11, out_G12 = bin_mnist.makeGeneratorLayers( MINIBATCH_SIZE, Z, Z_SIZE, Y, NUM_Y)
gX = ll.get_output(gX_layers, deterministic=False)

print 'getGenParams:'
gen_params, gen_sp_params = bin_mnist.getGenParams()

## Discriminator(D)
D_layers, layer_X, layer_Y = bin_mnist.makeDiscriminator( MINIBATCH_SIZE, X, (MINIBATCH_SIZE, nc, npx, npx), Y, NUM_Y )
# D output for Real Data
p_real = ll.get_output(D_layers, inputs={layer_X:X})
# D output for Generated Data
p_gen  = ll.get_output(D_layers, inputs={layer_X:gX})

print 'getDisParams:'
discrim_params, discrim_sp_params = bin_mnist.getDisParams()

## Costs

# Cost function of D for real data = average of BCE(binary cross entropy)
d_cost_real = lo.binary_crossentropy(p_real, T.ones(p_real.shape)).mean()

# Cost function of D for gen  data = average of BCE
d_cost_gen = lo.binary_crossentropy(p_gen, T.zeros(p_gen.shape)).mean()

# Const function of G = average of BCE
g_cost_d = lo.binary_crossentropy(p_gen, T.ones(p_gen.shape)).mean()

# total cost of D
d_cost = d_cost_real + d_cost_gen

# total cost of G
g_cost = g_cost_d

# total costs
cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

#
# Updater
#
lrt = sharedX(lr)

d_updates = lu.adam(d_cost, discrim_params, learning_rate=lrt, beta1=b1)

#
g_updates=None
if IS_BINARY:
    g_updates = bin_mnist.getGeneratorUpdates(loss=g_cost, aLearningRate=lrt, aBeta1=b1)
else:
    g_updates = lu.adam(g_cost, gen_params, learning_rate=lrt, beta1=b1)

#updates = d_updates + g_updates
updates = OrderedDict(d_updates.items() + g_updates.items())

#
# training function
#
sys.stdout.flush()
print( 'COMPILING...' )
t = time()
_train_g = theano.function([X, Z, Y], cost, updates=g_updates)
_train_d = theano.function([X, Z, Y], cost, updates=d_updates)
_gen = theano.function([Z, Y], [gX, out_lYS, out_G3_1, out_G3_2, out_G10, out_G11, out_G12] )
print( 'COMPILING...DONE' )
print( '%.2f seconds to compile theano functions'%(time()-t))

#
# Saving the training images in grey-scaled.
#
tr_idxs = np.arange(len(trX0))
trX0_vis = np.asarray([
    [
        trX0[i] for i in py_rng.sample(tr_idxs[trY0==y], 20)
    ] for y in range(10)
]).reshape(200, -1)

trX0_vis = inverse_transform(transform(trX0_vis))
grayscale_grid_vis(trX0_vis, (10, 20), 'samples/%s_etl_test.png'%desc)

f_log = open('logs/%s.ndjson'%desc, 'wb')
log_fields = [
    'n_epochs',
    'n_updates',
    'n_examples',
    'n_seconds',
    '1k_va_nnc_acc',
    '10k_va_nnc_acc',
    '100k_va_nnc_acc',
    '1k_va_nnd',
    '10k_va_nnd',
    '100k_va_nnd',
    'g_cost',
    'd_cost',
]

n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()

sample_zmb = sample_z(MINIBATCH_SIZE, Z_SIZE)
tmp_ymb = []
for ii in range(0, MINIBATCH_SIZE):
    tmp_ymb.append(ii % 10)

sample_ymb = reshapeY_forBNN(np.array(tmp_ymb))

sys.stdout.flush()

#
# Training loop (main loop)
#
at_first = True
for epoch in range(1, niter+niter_decay+1):
    # shuffle
    trX0, trX, trY = shuffle(trX0, trX, trY)

    # Generate images at this time.
    genout, out_lYS, out_G3_1, out_G3_2, out_G10, out_G11, out_G12 = _gen(sample_zmb, sample_ymb)
    samples = genout
    grayscale_grid_vis(inverse_transform(samples), (10, 10), 'samples/%s/%d.png'%(desc, n_epochs))

    #
    for imb0, imb, ymb in tqdm(iter_data(trX0, trX, trY, size=MINIBATCH_SIZE), total=ntrain/MINIBATCH_SIZE):
        # X:real data
        if not IS_BINARY:
            # transform imb to (?, 1, 28, 28)
            imb = transform(imb)
            ymb = floatX(np.uint8(OneHot(ymb, NUM_Y)))

        #imb:[0.0, 255]
        imb = expandRows( imb, MINIBATCH_SIZE )
        if at_first is True:
            print 'imb:', imb.shape, np.min(imb), np.max(imb)

        # Y: label
        ymb = expandRows( ymb, MINIBATCH_SIZE )
        if at_first is True:
            print 'ymb:', ymb.shape, np.min(ymb), np.max(ymb)

        # Z: random variabel from Gaussian
        zmb = sample_z(len(imb), Z_SIZE)
        if at_first is True:
            print_out(zmb, 'zmb')
            at_first = False

        # Train G and D each other
        if n_updates % (k+1) == 0:
            # Train G
            cost = _train_g(imb, zmb, ymb)
        else:
            # Train D
            cost = _train_d(imb, zmb, ymb)

        n_updates += 1
        n_examples += len(imb)

    n_epochs += 1
    if n_epochs > niter:
        # Update the Learning Rate
        if LR_DECAY_ALGO==1:
            lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
        elif LR_DECAY_ALGO==2:
            lrt.set_value(floatX(lr2))

        print 'Learning Late=', lrt.get_value()

        joblib.dump([p for p in gen_params], 'models/%s/%04d_gen_params.jl'%(desc, n_epochs), compress=True)
        joblib.dump([p for p in gen_sp_params], 'models/%s/%04d_gen_sp_params.jl'%(desc, n_epochs), compress=True)
        #joblib.dump([p for p in discrim_params], 'models/%s/%04d_discrim_params.jl'%(desc, n_epochs), compress=True)
