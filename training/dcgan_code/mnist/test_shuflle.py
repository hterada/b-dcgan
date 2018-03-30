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
from lib.ops import batchnorm, conv_cond_concat, conv_cond_concat2, deconv, dropout
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data
from lib.metrics import nnc_score, nnd_score

from load import mnist_with_valid_set

def printVal(name, V):
    print '%s:' % name, type(V), 'ndim:', V.ndim, ':', V

# Y はラベル（Conditional GAN の方式）
def gen(Z, Y, w, w2, w3, wx):
    #Z: (nbatch, nz) = (128, 100)
    #Y: (nbatch, ny) = (128, 10)
    #w: (nz+ny, ngfc) = (110, 1024)
    #w2: (ngfc+ny, ngf*2*7*7) = (1024+10, 64*2*7*7) = (1034, 6272)
    #w3: (ngf*2+ny, ngf, 5, 5) = (128+10, 64, 5, 5 ) = (138, 64, 5, 5)
    #wx: (ngf+ny, nc, 5, 5) = (64+10, 1, 5, 5) = (74, 1, 5, 5)
    print '\n@@@@ gen()'
    printVal( 'Y', Y )
    printVal( 'w', w )   #matrix
    printVal( 'w2', w2 ) #matrix
    printVal( 'w3', w3 ) #tensor
    printVal( 'wx', wx ) #tensor
    # Yの要素の並びの入れ替え。数字の引数は、次元番号。'x' は ブロードキャスト
    #(G1)
    yb = Y.dimshuffle(0, 1, 'x', 'x')
    # yb は４次元テンソル
    printVal('yb', yb)
    # 行列 Z と Y を結合（横方向）:いわゆる Conditional GAN の形にする。

    #(G2)
    Z = T.concatenate([Z, Y], axis=1) # Z: (128, 110)

    #(G3)
    # Z*w をバッチ正規化して、ReLU 適用
    t1 = T.dot(Z, w) #full connect : t1: (128, 1024)
    printVal('t1', t1)
    h = relu(
        batchnorm(
            t1
        )
    )
    # h: (128, 1024)
    #(G4)
    h = T.concatenate([h, Y], axis=1) # h: (128, 1034)
    #(G5)
    h2 = relu(
        batchnorm(
            T.dot(h, w2) #NOT full connect
        )
    )
    #(G6)
    h2 = h2.reshape((h2.shape[0], ngf*2, 7, 7))

    #(G7)
    h3, yb2 = conv_cond_concat2(h2, yb) #XXX

    #(G8)デコンボリューション:論文によれば、空間プーリングの代わりに適用する
    d = deconv(h3, w3, subsample=(2, 2), border_mode=(2, 2))
    printVal( 'd', d ) # (128, 64, 14, 14)
    #h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))
    #(G9)
    h4 = relu(
        batchnorm(d)
    )
    #(G10)
    h5, yb3 = conv_cond_concat2(h4, yb)
    #(G11)
    x = sigmoid(
        deconv(h5, wx, subsample=(2, 2), border_mode=(2, 2)
        )
    )
    return x, yb, yb2, d, h3, h5

def discrim(X, Y, w, w2, w3, wy):
    yb = Y.dimshuffle(0, 1, 'x', 'x')
    printVal('w', w )
    printVal('w2', w2)
    printVal('w3', w3 )
    printVal('wy', wy )
    X = conv_cond_concat(X, yb)
    h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
    h = conv_cond_concat(h, yb)
    h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2))))
    h2 = T.flatten(h2, 2)
    h2 = T.concatenate([h2, Y], axis=1)
    h3 = lrelu(batchnorm(T.dot(h2, w3)))
    h3 = T.concatenate([h3, Y], axis=1)
    y = sigmoid(T.dot(h3, wy))
    return y

def transform(X):
    return (floatX(X)/255.).reshape(-1, nc, npx, npx)

def inverse_transform(X):
    X = X.reshape(-1, npx, npx)
    return X

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
ny = 10           # # of classes
nbatch = 128      # # of examples in batch
npx = 28          # # of pixels width/height of images
nz = 100          # # of dim for Z
ngfc = 1024       # # of gen units for fully connected layers
ndfc = 1024       # # of discrim units for fully connected layers
ngf = 64          # # of gen filters in first conv layer
ndf = 64          # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 100       # # of iter at starting learning rate
niter_decay = 100 # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
ntrain, nval, ntest = len(trX), len(vaX), len(teX)

desc = 'cond_dcgan'
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

# 正規分布 Normal Distribution
gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)

# 正規分布によるランダム値(shape, name)
gw  = gifn((nz+ny, ngfc), 'gw')

gw2 = gifn((ngfc+ny, ngf*2*7*7), 'gw2') #**

gw3 = gifn((ngf*2+ny, ngf, 5, 5), 'gw3')
gwx = gifn((ngf+ny, nc, 5, 5), 'gwx')

# 正規分布によるランダム値(shape, name)
dw  = difn((ndf, nc+ny, 5, 5), 'dw')
dw2 = difn((ndf*2, ndf+ny, 5, 5), 'dw2')
dw3 = difn((ndf*2*7*7+ny, ndfc), 'dw3')
dwy = difn((ndfc+ny, 1), 'dwy')

gen_params = [gw, gw2, gw3, gwx]
discrim_params = [dw, dw2, dw3, dwy]


X = T.tensor4()
Z = T.matrix()
Y = T.matrix()

# ジェネレータのモデル
gX, yb, yb2, d, h3, h5 = gen(Z, Y, *gen_params)
print('gX', type(gX), gX)

# ディスクリミネータのモデル
## 実データについて
p_real = discrim(X, Y, *discrim_params)
## ジェネレータが生成したデータについて
p_gen = discrim(gX, Y, *discrim_params)

# bce: バイナリクロスエントロピー
# Dのコスト関数（実データについての）＝BCEの平均値
d_cost_real = bce(p_real, T.ones(p_real.shape)).mean()
# Dのコスト関数（Gデータについての）＝BCEの平均値
d_cost_gen = bce(p_gen, T.zeros(p_gen.shape)).mean()
# Gのコスト関数（Gデータ）＝BCEの平均値
g_cost_d = bce(p_gen, T.ones(p_gen.shape)).mean()

# Dの総合コスト
d_cost = d_cost_real + d_cost_gen
# Gのコスト
g_cost = g_cost_d

# コストのまとめ
cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
updates = d_updates + g_updates

print 'COMPILING'
t = time()
_train_g = theano.function([X, Z, Y], cost, updates=g_updates)
_train_d = theano.function([X, Z, Y], cost, updates=d_updates)

###
_gen = theano.function([Z, Y], [gX, yb, yb2, d, h3, h5])
print '%.2f seconds to compile theano functions'%(time()-t)

tr_idxs = np.arange(len(trX))
trX_vis = np.asarray([[trX[i] for i in py_rng.sample(tr_idxs[trY==y], 20)] for y in range(10)]).reshape(200, -1)
trX_vis = inverse_transform(transform(trX_vis))
grayscale_grid_vis(trX_vis, (10, 20), 'samples/%s_etl_test.png'%desc)

sample_zmb = floatX(np_rng.uniform(-1., 1., size=(200, nz)))
sample_ymb = floatX(OneHot(np.asarray([[i for _ in range(20)] for i in range(10)]).flatten(), ny))

def gen_samples(n, nbatch=128):
    samples = []
    labels = []
    n_gen = 0
    for i in range(n/nbatch):
        ymb = floatX(OneHot(np_rng.randint(0, 10, nbatch), ny))
        zmb = floatX(np_rng.uniform(-1., 1., size=(nbatch, nz)))
        xmb, tmp_yb, yb2, d, h3, h5 = _gen(zmb, ymb)
        print 'tmp_yb:', tmp_yb.shape
        print 'yb2:', yb2.shape
        print 'd:', d.shape
        print 'h3:', h3.shape
        print 'h5:', h5.shape
        sys.exit()
        samples.append(xmb)
        labels.append(np.argmax(ymb, axis=1))
        n_gen += len(xmb)
    n_left = n-n_gen
    ymb = floatX(OneHot(np_rng.randint(0, 10, n_left), ny))
    zmb = floatX(np_rng.uniform(-1., 1., size=(n_left, nz)))
    xmb = _gen(zmb, ymb)
    samples.append(xmb)
    labels.append(np.argmax(ymb, axis=1))
    return np.concatenate(samples, axis=0), np.concatenate(labels, axis=0)

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

print desc.upper()
n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()

gX, gY = gen_samples(100000)
