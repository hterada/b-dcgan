# -*- coding: utf-8 -*-
#
# Binary-DCGAN
# by H.terada / OpenStream, Inc.
#

import sys
#sys.path.append('..')
sys.path.append('dcgan_code')
#sys.path.append('dcgan_code/lib')
sys.path.append('dcgan_code/mnist')
#sys.path.append('BinaryNet/Train_time')

import os
import time
import inspect
from itertools import chain

import math
import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1')
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne
import lasagne.layers as ll
import lasagne.nonlinearities as ln

import BinaryNet.Train_time.binary_net as binary_net
import binary_net_ex
from BinaryNet.Train_time.binary_net import binarization
from BinaryNet.Train_time.binary_net import round3

from collections import OrderedDict

from dcgan_code.lib import inits
from dcgan_code.lib import updates
from dcgan_code.lib.theano_utils import floatX, sharedX
from dcgan_code.lib.rng import py_rng, np_rng

from dcgan_code.lib.data_utils import OneHot, shuffle, iter_data

from dcgan_code.mnist.load import mnist_with_valid_set

# BatchNorm params
ALPHA = .1
EPSILON = 1e-4

#
IS_STOCHASTIC = False
H = 1.0
W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
NUM_IMG_CHANNELS = 1

NUM_DIS_FC_UNITS = 1024
NUM_DIS_FILTERS = 64

#
bce = T.nnet.binary_crossentropy

def nop_func(X):
    return X

class BinaryMnist(object):
    def __init__(self, is_binary):
        self.IS_BINARY = is_binary

        self.NUM_FC_UNITS = 600
        self.NUM_GEN_FILTERS = 64

        ### Scenario Parameters

        # Encoder
        self.NUM_HIDDEN_LAYERS = 3
        self.IS_INPUT_AS_INT    = True
        if self.IS_INPUT_AS_INT:
            h=2
            self.A=math.pow(2,(h-1))-1
        else:
            self.A=1.0

        self.IS_USE_B_FC     = True
        self.IS_USE_B_BNA_1  = True

        # Decoder
        self.IS_USE_B_DECONV_1  = True
        self.IS_USE_B_DECONV_2  = False
        self.IS_USE_B_BNA_2     = True

        self.IS_DIS_BIN = False #experimental

    def getVersionDesc(self):
        return 'BinaryMnist ver2; Z input as int'

    def getProperties(self):
        attrs = inspect.getmembers(self, lambda a: not(inspect.isroutine(a)))
        props = filter(lambda a: not(a[0].startswith('__')), attrs)
        for prop in props:
            print 'prop:', prop

    def makeGeneratorLayers(self, aNBatch, aZ, aZSize, aY, aYSize):
        '''
        aZ: input for NN (random, shape=(nbatch, ?)
        aY: label (condition, shape=(nbatch, ?) ) ; [-1, +1]
        '''

        pre_func = nop_func
        if self.IS_INPUT_AS_INT:
            pre_func = round3

        #(G1)
        yb = pre_func(aY.dimshuffle(0, 1, 'x', 'x'))

        # Input layer
        layer_Z = ll.InputLayer(shape=(aNBatch, aZSize), input_var=pre_func(aZ*self.A), name='Z')
        layer_Y = ll.InputLayer(shape=(aNBatch, aYSize), input_var=aY, name='Y')
        layer_YGH = ll.InputLayer(shape=(aNBatch, aYSize), input_var=pre_func(aY*self.A), name='YGH')
        out_lYS = ll.get_output(layer_Y)

        # encoder
        gen, out_G3_1, out_G3_2 = self.makeGenerator_encoder( layer_Z, layer_Y, layer_YGH )

        # decoder
        gen, out_G10, out_G11, out_G12 = self.makeGenerator_decoder( gen, yb, aYSize )

        self.gen = gen
        return gen, out_lYS, out_G3_1, out_G3_2, out_G10, out_G11, out_G12

    def makeGenerator_encoder(self, layer_Z, layer_Y, layer_YGH):
        #(G2)
        # as the Conditional GAN style
        gen = ll.ConcatLayer( [layer_Z, layer_YGH], axis=1 )

        #(G3)
        #(G3-1) Full Connect (w)
        out_G3_1=None
        for k in range(self.NUM_HIDDEN_LAYERS):
            if self.IS_USE_B_FC:
                gen = binary_net.DenseLayer(
                    gen,
                    binary=True,
                    stochastic=IS_STOCHASTIC,
                    H=H,
                    W_LR_scale=W_LR_scale,
                    b=None, #No Bias
                    nonlinearity=None,
                    num_units=self.NUM_FC_UNITS
                )
            else:
                gen = ll.DenseLayer(
                    gen,
                    num_units=self.NUM_FC_UNITS,
                    nonlinearity=None
                )

            print 'G3-1:gen.shape', gen.input_shape, gen.output_shape
            if out_G3_1 is None:
                out_G3_1 = ll.get_output(gen)

            #(G3-2) Batch Norm
            if self.IS_USE_B_BNA_1:
                # This layer includes the activation process
                gen = binary_net_ex.BatchNormLayer( gen, epsilon=EPSILON, alpha=ALPHA, H=1 )
                print 'G3-2:gen.shape', gen.input_shape, gen.output_shape
            else:
                gen = ll.BatchNormLayer( gen, epsilon=EPSILON, alpha=ALPHA )
                print 'G3-2:gen.shape', gen.input_shape, gen.output_shape

                #(G3-3) Activation: Binary tanh
                gen = ll.NonlinearityLayer( gen, nonlinearity=binary_net.binary_tanh_unit )
                print 'G3-3:gen.shape', gen.input_shape, gen.output_shape

            out_G3_2 = ll.get_output(gen)

            #END for

        #(G4) Concat
        gen = ll.ConcatLayer( [gen, layer_Y], axis=1 )

        #(G5)
        #(G5-1) Full connect (w2)
        if self.IS_USE_B_FC:
            gen = binary_net.DenseLayer(
                gen,
                binary=True,
                stochastic=IS_STOCHASTIC,
                H=H,
                W_LR_scale=W_LR_scale,
                b=None, #No Bias
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=(self.NUM_GEN_FILTERS*7*7)
            )
        else:
            gen = ll.DenseLayer(
                gen,
                num_units=(self.NUM_GEN_FILTERS*7*7),
                nonlinearity=None
            )

        print 'G5-1:gen.shape', gen.input_shape, gen.output_shape #(128,3136)

        #(G5-2) Batch Norm
        if self.IS_USE_B_BNA_1:
            gen = binary_net_ex.BatchNormLayer( gen, epsilon=EPSILON, alpha=ALPHA, H=1 )
        else:
            gen = ll.BatchNormLayer( gen, epsilon=EPSILON, alpha=ALPHA )

            #(G5-3) Activation: Binary tanh
            gen = ll.NonlinearityLayer( gen, nonlinearity=binary_net.binary_tanh_unit )
            print 'G5-3:gen.shape', gen.input_shape, gen.output_shape

        #(G6) Reshape
        gen = ll.ReshapeLayer(
            gen,
            # shape [0] denoting to use the size of the 0-th input dimension
            shape=([0], self.NUM_GEN_FILTERS, 7, 7) #TODO constat var.
        )

        return gen, out_G3_1, out_G3_2

    def makeGenerator_decoder(self, gen, yb, aYSize ):
        #(G7)
        gen_dec = self.conv_cond_concat( gen, yb, aYSize )
        print 'G7:', gen_dec.output_shape #(128, 74, 7, 7)

        #(G8)
        # deconvolution
        if self.IS_USE_B_DECONV_1:
            gen_dec = binary_net_ex.Deconv2DLayer( gen_dec, num_filters = self.NUM_GEN_FILTERS,
                                                   filter_size=(5,5),
                                                   stride=(2,2),
                                                   output_size=(14,14),
                                                   crop=2,
                                                   nonlinearity=None,
                                                   binary=True,
                                                   stochastic=IS_STOCHASTIC,
                                                   H=H,
                                                   W_LR_scale=W_LR_scale,
                                                   b=None
            )
        else:
            gen_dec = ll.Deconv2DLayer( gen_dec, num_filters = self.NUM_GEN_FILTERS,
                                        filter_size=(5,5),
                                        stride=(2,2),
                                        output_size=(14,14),
                                        b=None,
                                        crop=2,
                                        nonlinearity=None
            )
        print 'G8:gen_dec.shape', gen_dec.input_shape, gen_dec.output_shape #(128, 64, 14, 14)

        #(G9)
        if self.IS_USE_B_BNA_2:
            gen_dec = binary_net_ex.BatchNormLayer( gen_dec, verbose=True, epsilon=EPSILON, alpha=ALPHA, H=1.0 )
        else:
            gen_dec = ll.BatchNormLayer( gen_dec, epsilon=EPSILON, alpha=ALPHA )
            gen_dec = ll.NonlinearityLayer( gen_dec, nonlinearity=binary_net.binary_tanh_unit )
        print 'G9:gen_dec.shape', gen_dec.input_shape, gen_dec.output_shape

        #(G10)
        gen_dec = self.conv_cond_concat( gen_dec, yb, aYSize )
        print 'G10:gen_dec.shape', gen_dec.output_shape
        out_G10 = ll.get_output(gen_dec)

        #(G11)
        if self.IS_USE_B_DECONV_2:
            gen_dec = binary_net_ex.Deconv2DLayer( gen_dec, num_filters = NUM_IMG_CHANNELS,
                                                   filter_size=(5,5), stride=(2,2),
                                                   output_size=(28,28),
                                                   crop=2,
                                                   nonlinearity=None,
                                                   binary=True,
                                                   stochastic=IS_STOCHASTIC,
                                                   H=H,
                                                   W_LR_scale=W_LR_scale,
                                                   b=None
            )
        else:
            gen_dec = ll.Deconv2DLayer( gen_dec, num_filters = NUM_IMG_CHANNELS,
                                        filter_size=(5,5), stride=(2,2),
                                        output_size=(28,28),
                                        b=None,
                                        crop=2,
                                        nonlinearity=None
            )

        print 'G11:gen_dec.shape', gen_dec.input_shape, gen_dec.output_shape
        out_G11 = ll.get_output(gen_dec)

        #(G12)
        if self.IS_USE_B_BNA_2:
            #gen_dec = ll.ExpressionLayer(gen_dec, function = lambda X : X/self.A)
            pass
        #
        gen_dec = ll.NonlinearityLayer( gen_dec, nonlinearity=lasagne.nonlinearities.sigmoid ) #TODO binary ?

        print 'G12:gen_dec.shape', gen_dec.input_shape, gen_dec.output_shape
        out_G12 = ll.get_output(gen_dec)

        return gen_dec, out_G10, out_G11, out_G12

    def getGeneratorUpdates(self, loss, aLearningRate, aBeta1):
        LR=aLearningRate
        mlp = self.gen
        if self.IS_BINARY:
            # W updates
            # extract 'binary' parameters only
            Wb_list = lasagne.layers.get_all_params(self.gen, binary=True)
            for eW in Wb_list:
                #print( 'eW:', type(eW), eW )
                pass

            # Make a list of the gradients w.r.t. the binary parameters
            W_grad_list = binary_net.compute_grads(loss, mlp)
            #print('W_grad_list', type(W_grad_list), W_grad_list)

            # Update function map(OrderedDict) with ADAM learning method
            updates_b0 = lasagne.updates.adam(loss_or_grads=W_grad_list, params=Wb_list, learning_rate=LR,
                                              beta1=aBeta1
            )

            # clipping & scaling for binarization
            updates_b1 = binary_net.clipping_scaling(updates_b0, mlp)

            # other parameters updates
            Wr_list = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)
            for Wr in Wr_list:
                #print('Wr:', type(Wr), Wr)
                pass

            # Marging the parameters : binary params + other params
            updates = OrderedDict(updates_b1.items() +
                                  lasagne.updates.adam(loss_or_grads=loss,
                                                       params=Wr_list, learning_rate=LR, beta1=aBeta1).items())

        else:
            Wr_list = lasagne.layers.get_all_params(mlp, trainable=True)
            updates = lasagne.updates.adam(loss_or_grads=loss, params=Wr_list, learning_rate=LR, beta1=aBeta1)

        return updates

    def makeDiscriminator(self, aNBatch, aX, aXShape, aY, aYSize):
        #(D1)
        yb = aY.dimshuffle(0, 1, 'x', 'x')

        #(D2)
        layer_X = ll.InputLayer(shape=aXShape, input_var=aX, name='lX')
        layer_Y = ll.InputLayer(shape=(aNBatch, aYSize), input_var=aY, name='lY')
        dis = self.conv_cond_concat(layer_X, yb, aYSize)

        #(D3), (D4)
        if self.IS_DIS_BIN:
            dis = binary_net_ex.Conv2DLayer(dis,
                                            num_filters = NUM_DIS_FILTERS,
                                            filter_size = (5,5),
                                            stride = (2, 2),
                                            nonlinearity=ln.LeakyRectify(0.2), #TODO
                                            pad=2,
                                            binary=True,
                                            stochastic=IS_STOCHASTIC,
                                            H=H,
                                            W_LR_scale=W_LR_scale
            )
        else:
            dis = ll.Conv2DLayer(dis,
                                 num_filters = NUM_DIS_FILTERS,
                                 filter_size = (5,5),
                                 stride = (2, 2),
                                 nonlinearity=ln.LeakyRectify(0.2),
                                 pad=2)
        print 'D4:', dis.output_shape # (128, 64, 14, 14)

        #(D5)
        dis = self.conv_cond_concat(dis, yb, aYSize)

        #(D6)
        if self.IS_DIS_BIN:
            dis = binary_net_ex.Conv2DLayer(dis,
                                            num_filters = NUM_DIS_FILTERS*2,
                                            filter_size = (5, 5),
                                            stride = (2, 2),
                                            nonlinearity=None,
                                            pad=2,
                                            binary=True,
                                            stochastic=IS_STOCHASTIC,
                                            H=H,
                                            W_LR_scale=W_LR_scale
            )
        else:
            dis = ll.Conv2DLayer(dis,
                                 num_filters = NUM_DIS_FILTERS*2,
                                 filter_size = (5, 5),
                                 stride = (2, 2),
                                 nonlinearity=None,
                                 pad=2)
        print 'D6:', dis.output_shape # (128, 128, 7, 7)

        dis = ll.BatchNormLayer(dis, epsilon=EPSILON, alpha=ALPHA)
        dis = ll.NonlinearityLayer(dis, nonlinearity=ln.LeakyRectify(0.2)) #TODO
        #(D7)
        dis = ll.FlattenLayer(dis, outdim=2)
        print 'D7:', dis.output_shape # (128, 6272)
        #(D8)
        dis = ll.ConcatLayer([dis, layer_Y], axis=1)
        #(D9)
        if self.IS_DIS_BIN:
            dis = binary_net_ex.DenseLayer(dis, num_units=NUM_DIS_FC_UNITS,
                                           binary=True,
                                           stochastic=IS_STOCHASTIC,
                                           H=H,
                                           W_LR_scale=W_LR_scale,
                                           b=None, #No Bias
                                           nonlinearity=None
            )
        else:
            dis = ll.DenseLayer(dis, num_units=NUM_DIS_FC_UNITS)

        dis = ll.BatchNormLayer(dis, epsilon=EPSILON, alpha=ALPHA)
        dis = ll.NonlinearityLayer(dis, nonlinearity=ln.LeakyRectify(0.2)) #TODO
        #(D10)
        dis = ll.ConcatLayer([dis, layer_Y], axis=1)


        #(D11) OUTPUT layer
        dis = ll.DenseLayer(dis, num_units=1,
                            nonlinearity=ln.sigmoid)
        print 'D11:', dis.output_shape # (128, 1)

        self.dis = dis
        return dis, layer_X, layer_Y


    def getGenParams(self):
        return self.getParamsOf(self.gen)

    def getDisParams(self):
        return self.getParamsOf(self.dis)

    def getParamsOf(self, layers):
        print '== getParmasOf()'
        ls = ll.get_all_layers(layers)
        for ly in ls:
            #print(ly.name, 'keys:', ly.params.keys())
            pass
        params = [] #list

        # binary params
        bin_params      = ll.get_all_params(layers, binary=True)
        print '%s bin_params len:%d' % (hex(id(layers)), len(bin_params))
        for bp in bin_params:
            print 'bp:', type(bp), bp.name, bp.shape.eval()

        # real value params
        rv_params       = ll.get_all_params(layers, trainable=True, binary=False)
        print '%s  rv_params len:%d' % (hex(id(layers)), len(rv_params))
        for rp in rv_params:
            print 'rp:', type(rp), rp.name, rp.shape.eval()

        # special params
        sp_params       = ll.get_all_params(layers, trainable=False, binary=False)
        print '%s  sp_params len:%d' % (hex(id(layers)), len(sp_params))
        for sp in sp_params:
            print 'sp:', type(sp), sp.name, sp.shape.eval()

        params.extend(bin_params)
        params.extend(rv_params)
        return params, sp_params

    def conv_cond_concat(self, aLayer, aY, aYSize):
        if True:
            '''
            aY: theano var.
            '''
            x_shape = aLayer.output_shape
            oned = T.ones( (x_shape[0], aYSize, x_shape[2], x_shape[3]) )
            y_oned = aY*oned # [Ymin, Ymax]
            l_aY = ll.InputLayer(input_var=y_oned, shape=(x_shape[0], aYSize, x_shape[2], x_shape[3]))
            layer = ll.ConcatLayer([aLayer, l_aY],
                                   axis=1)
            return layer

        else:
            return aLayer

#
#
# main
#
#

if __name__ == '__main__':
    NUM_CLASSES = 10           # # of classes
    nz = 100          # # of dim for Z

    Z = T.matrix('random')
    Y = T.matrix('label')

    #####
    mnist = BinaryMnist()
    generator = mnist.makeGeneratorLayers(NUM_MINIBATCH, Z, nz, Y, NUM_CLASSES)
    out = ll.get_output(generator)

    print 'compiling...'
    out_func = theano.function([Z, Y], out, mode='DebugMode')
    print 'compiling...DONE'

    #test
    Zval = floatX(np_rng.uniform(-1.0, 1.0, size=(NUM_MINIBATCH, nz)))
    Yval = floatX(OneHot(np_rng.randint(0, 10, NUM_MINIBATCH), NUM_CLASSES))

    ret = out_func(Zval,Yval)
    print 'ret', ret.shape
