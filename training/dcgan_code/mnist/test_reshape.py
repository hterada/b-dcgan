# -*- coding:utf-8 -*-
import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from tqdm import tqdm #プログレスバー用
from matplotlib import pyplot as plt
from sklearn.externals import joblib

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

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

def printVal(name, V):
    print '%s:' % name, type(V), 'ndim:', V.ndim, ':', V

def gen(Z, Y, w, w2, w3, wx):
    print '\n@@@@ gen()'
    printVal( 'Z', Z )  # matrix
    #printVal( 'Y', Y )  # matrix
    printVal( 'w', w )  # matrix
    printVal( 'w2', w2 )# matrix

    printVal( 'w3', w3 )# tensor
    printVal( 'wx', wx )# tensor
    # Yの要素の並びの入れ替え。数字の引数は、次元番号。'x' は ブロードキャスト
    # 並び替えの前後で、全体の要素数は変わらない。
    yb = Y.dimshuffle(0, 1, 'x', 'x')
    # yb は４次元テンソル
    #printVal('yb', yb)
    # 行列 Z と Y を結合（横方向）
    Z = T.concatenate([Z, Y], axis=1) # matrix
    # Z*w(Full Connect) をバッチ正規化して、ReLU 適用
    tmp_a = T.dot(Z,w) # dot(matrix, matrix)->matrix
    printVal( 'dot(Z,w) -> tmp_a', tmp_a )
    h = relu(batchnorm(T.dot(Z, w))) #CCC
    h = T.concatenate([h, Y], axis=1) #CCC

    printVal('h', h)    # matrix
    h2 = relu(batchnorm(T.dot(h, w2))) #CCC
    printVal( 'h2', h2) #h2:matrix
    h2r = h2.reshape((h2.shape[0], GEN_NUM_FILTER*2, 7, 7)) #CCC
    printVal( 'h2r', h2r) #h2r:tensor
    h2ry = conv_cond_concat(h2r, yb) #
    printVal( 'h2ry', h2ry ) #h2:tensor
    # デコンボリューション:論文によれば、空間プーリングの代わりに適用する
    d = deconv(h2ry, w3, subsample=(2, 2), border_mode=(2, 2))
    printVal( 'd', d )
    #h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))
    h3 = relu(batchnorm(d))
    h3 = conv_cond_concat(h3, yb)
    x = sigmoid(deconv(h3, wx, subsample=(2, 2), border_mode=(2, 2)))
    return x, h2

#
#
# main
#
#

# MNISTデータをロード
#  trX:トレーニング画像(28x28)
#  vaX:バリデーション画像（シャッフル後の trX の 50000<=index のデータ）
#  teX:テスト画像
#  trY:トレーニングラベル（シャッフル後の trY の index<50000 のデータ )
#  vaY:バリデーションラベル(シャッフル後の trY の 50000<=index のデータ)
#  teY:テストラベル
trX, vaX, teX, trY, vaY, teY = mnist_with_valid_set()

# vaX を 0.0--1.0 に変換
vaX = floatX(vaX)/255.

k = 1             # # of discrim updates for each gen update
l2 = 2.5e-5       # l2 weight decay
b1 = 0.5          # momentum term of adam
nc = 1            # # of channels in image
NUM_LABEL = 10           # # of classes
nbatch = 128      # # of examples in batch
npx = 28          # # of pixels width/height of images
DIM_Z = 100          # # of dim for Z
GEN_NUM_FC = 1024       # # of gen units for fully connected layers
ndfc = 1024       # # of discrim units for fully connected layers
GEN_NUM_FILTER = 64          # # of gen filters in first conv layer
ndf = 64          # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 100       # # of iter at starting learning rate
niter_decay = 100 # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam

desc = 'cond_dcgan'
samples_dir = 'samples/%s'%desc

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
bce = T.nnet.binary_crossentropy

# 正規分布 Normal Distribution
gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)

# 正規分布によるランダム値(shape, name)
gw  = gifn((DIM_Z+NUM_LABEL, GEN_NUM_FC), 'gw')
gw2 = gifn((GEN_NUM_FC+NUM_LABEL, GEN_NUM_FILTER*2*7*7), 'gw2')
gw3 = gifn((GEN_NUM_FILTER*2+NUM_LABEL, GEN_NUM_FILTER, 5, 5), 'gw3')        #conv?
gwx = gifn((GEN_NUM_FILTER+NUM_LABEL, nc, 5, 5), 'gwx')           #conv?

# 正規分布によるランダム値(shape, name)
dw  = difn((ndf, nc+NUM_LABEL, 5, 5), 'dw')
dw2 = difn((ndf*2, ndf+NUM_LABEL, 5, 5), 'dw2')
dw3 = difn((ndf*2*7*7+NUM_LABEL, ndfc), 'dw3')
dwy = difn((ndfc+NUM_LABEL, 1), 'dwy')

gen_params = [gw, gw2, gw3, gwx]
discrim_params = [dw, dw2, dw3, dwy]


X = T.tensor4()
Z = T.matrix()
Y = T.matrix()

# ジェネレータのモデル
gX, rH2 = gen(Z, Y, *gen_params)

lrt = sharedX(lr)

print 'COMPILING'
t = time()
_gen = theano.function([Z, Y], gX)
print '%.2f seconds to compile theano functions'%(time()-t)

def gen_samples(n, nbatch=128):
    samples = []
    labels = []
    n_gen = 0
    for i in range(n/nbatch):
        print 'i:', i
        # ymb.shape = (nbatch, ny)
        ymb = floatX(OneHot(np_rng.randint(0, 10, nbatch), NUM_LABEL))
        print 'gen_samples: ymb:', ymb.shape
        print ymb

        # zmb.shape = (nbatch, DIM_Z)
        zmb = floatX(np_rng.uniform(-1., 1., size=(nbatch, DIM_Z)))
        print 'gen_samples: zmb:', zmb.shape
        print zmb

        # xmd
        xmb = _gen(zmb, ymb)
        print 'gen_samples: xmb:', xmb.shape
        print rH2

        samples.append(xmb)
        labels.append(np.argmax(ymb, axis=1))
        n_gen += len(xmb)
    n_left = n-n_gen
    ymb = floatX(OneHot(np_rng.randint(0, 10, n_left), NUM_LABEL))
    zmb = floatX(np_rng.uniform(-1., 1., size=(n_left, DIM_Z)))
    xmb = _gen(zmb, ymb)
    samples.append(xmb)
    labels.append(np.argmax(ymb, axis=1))
    return np.concatenate(samples, axis=0), np.concatenate(labels, axis=0)

#test
samples, labels = gen_samples(128)
