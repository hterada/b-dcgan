# -*- coding:utf-8 -*-
from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1')
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import binary_net

from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial

from collections import OrderedDict

if __name__ == "__main__":

    # BN(バッチノーマライゼーション) parameters
    batch_size = 100
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    # alpha = .15
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))

    # MLP(Multi Layer Perceptron 多層パーセプトロン) parameters
    num_units = 4096
    print("num_units = "+str(num_units))
    n_hidden_layers = 3
    print("n_hidden_layers = "+str(n_hidden_layers))

    # Training parameters
    num_epochs = 1000
    print("num_epochs = "+str(num_epochs))

    # Dropout parameters
    dropout_in = .2 # 0. means no dropout
    print("dropout_in = "+str(dropout_in))
    dropout_hidden = .5
    print("dropout_hidden = "+str(dropout_hidden))

    # BinaryOut(バイナリ出力関数)
    # binary_tanh_unit を使う
    activation = binary_net.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")
    # activation = binary_net.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")

    # BinaryConnect
    binary = True
    print("binary = "+str(binary))
    stochastic = False
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))

    # Decaying LR (Learning Rate:学習率）
    LR_start = .003
    print("LR_start = "+str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...

    save_path = "mnist_parameters.npz"
    print("save_path = "+str(save_path))

    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))

    print('Loading MNIST dataset...')

    train_set = MNIST(which_set= 'train', start=0, stop = 50000, center = False)
    valid_set = MNIST(which_set= 'train', start=50000, stop = 60000, center = False)
    test_set = MNIST(which_set= 'test', center = False)

    # bc01 format
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
    # 入力は、range [0.0,1.0]。これを２倍して１引くと range [-1.0, +1.0]になる
    train_set.X = 2* train_set.X.reshape(-1, 1, 28, 28) - 1.
    valid_set.X = 2* valid_set.X.reshape(-1, 1, 28, 28) - 1.
    test_set.X = 2* test_set.X.reshape(-1, 1, 28, 28) - 1.

    # flatten targets
    print('#### train_set.y(org):', train_set.y.shape, train_set.y[0][0])
    train_set.y = np.hstack(train_set.y)
    valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)
    print('#### train_set.y(flatten):', train_set.y.shape)

    # Onehot the targets
    ## np.eye()は、単位行列の作成関数, np.eye()[] は、One Hot ベクトルを返す。
    train_set.y = np.float32(np.eye(10)[train_set.y])
    valid_set.y = np.float32(np.eye(10)[valid_set.y])
    test_set.y = np.float32(np.eye(10)[test_set.y])

    # for hinge loss
    train_set.y = 2* train_set.y - 1.
    valid_set.y = 2* valid_set.y - 1.
    test_set.y = 2* test_set.y - 1.

    print('#### train_set.X:', train_set.X.shape)
    print('#### train_set.y:', train_set.y.shape)

    #
    #
    #

    print('Building the MLP...')

    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets') # ターゲット（教師）ラベル。One Hot ベクトルの列。
    LR = T.scalar('LR', dtype=theano.config.floatX)

    ######################################################################
    ######################################################################
    ######################################################################

    # 入力層
    mlp = lasagne.layers.InputLayer(
            shape=(None, 1, 28, 28),
            input_var=input)

    mlp = lasagne.layers.DropoutLayer(
            mlp,
            p=dropout_in)

    for k in range(n_hidden_layers):

        # lasagne の denselayer のGPU版
        mlp = binary_net.DenseLayer(
                mlp,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=num_units)

        mlp = lasagne.layers.BatchNormLayer(
                mlp,
                epsilon=epsilon,
                alpha=alpha)

        mlp = lasagne.layers.NonlinearityLayer(
                mlp,
                nonlinearity=activation)

        mlp = lasagne.layers.DropoutLayer(
                mlp,
                p=dropout_hidden)

    mlp = binary_net.DenseLayer(
                mlp,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10)

    mlp = lasagne.layers.BatchNormLayer(
            mlp,
            epsilon=epsilon,
            alpha=alpha)

    train_output = lasagne.layers.get_output(mlp, deterministic=False)

    # squared hinge loss
    # target は One Hot ベクトル（教師ラベル）。それと train_output との内積＝類似度を使用している。
    loss = T.mean(T.sqr(T.maximum(0.0,1.0 - target*train_output ))) #XXX

    #
    # update expression の構築
    #

    if binary:

        # W updates
        # binary なパラメータだけを取り出す
        Wb_list = lasagne.layers.get_all_params(mlp, binary=True)
        for eW in Wb_list:
            print( 'eW:', type(eW), eW )

        # binary なパラメータのみに対する勾配を求めてリストアップ
        W_grad_list = binary_net.compute_grads(loss, mlp)
        print('W_grad_list', type(W_grad_list), W_grad_list)

        # ADAM学習則による更新式マップ(OrderedDict)
        updates_b0 = lasagne.updates.adam(loss_or_grads=W_grad_list, params=Wb_list, learning_rate=LR)

        # バイナリ化のためのクリッピング＆スケーリング
        updates_b1 = binary_net.clipping_scaling(updates_b0, mlp)

        # other parameters updates
        # 非バイナリパラメータの更新則
        Wr_list = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)

        # バイナリ＋非バイナリ：パラメータ群をまとめる
        updates = OrderedDict(updates_b1.items() +
                              lasagne.updates.adam(loss_or_grads=loss,
                                                   params=Wr_list, learning_rate=LR).items())

    else:
        Wr_list = lasagne.layers.get_all_params(mlp, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=Wr_list, learning_rate=LR)

    #
    # test 用
    #

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)

    #
    # train 関数、validation 関数の作成
    #

    print( 'Theano Compiling...' )
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary)
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates) #XXX

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])
    print( 'Theano Compiling...DONE' )

    print('Training...')

    #
    # 訓練＆評価の実行
    #
    binary_net.train(
        train_fn, #XXX (input, target, LR) -> loss
        val_fn,
        mlp,
        batch_size,
        LR_start,LR_decay,
        num_epochs,
        train_set.X,
        train_set.y, #XXX target

        valid_set.X,
        valid_set.y,

        test_set.X,
        test_set.y,
        save_path,
        shuffle_parts)
