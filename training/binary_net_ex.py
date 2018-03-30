# -*- coding:utf-8 -*-
import time
import copy

from collections import OrderedDict

import numpy as np
import scipy.sparse as sp

import theano
import theano.tensor as T
import theano.compile.sharedvalue as tcs

import lasagne
import lasagne.init as init

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from binary_net import round3, hard_sigmoid, binarization

floatT = np.float32

def binary_tanh_unit2(x, H):
    return (2.*round3(hard_sigmoid(x))-1.)*H

def binary_sigmoid_unit255(x):
    return round3(hard_sigmoid(x))*255.0

# by H.Terada
def binarization2(W, Z, H):

    # if value is zero,  skip binarization

    # [-1,1] -> [0,1]
    Wb = T.switch( T.neq(Z, 1.0), T.switch(T.round(hard_sigmoid(W/H)), H, -H), 0.0)

    # print("det")
    #Wb = T.round(Wb)

    # 0 or 1 -> -1 or 1
    #Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)
    Wb = T.cast(Wb, theano.config.floatX)

    return Wb

# by Hideo Terada
class Deconv2DLayer(lasagne.layers.TransposedConv2DLayer):
    def __init__(self, incoming, num_filters, filter_size,
                 binary = True, stochastic = True, H=1.,W_LR_scale="Glorot",
                 W=None,
                 **kwargs):

        self.binary = binary
        self.stochastic = stochastic

        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.H = floatT(np.sqrt(1.5 / (num_inputs + num_units)))
            # print("H = "+str(self.H))

        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.W_LR_scale = floatT(1./np.sqrt(1.5 / (num_inputs + num_units)))
            # print("W_LR_scale = "+str(self.W_LR_scale))

        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))

        if self.binary:
            if W is None:
                super(Deconv2DLayer, self).__init__(incoming, num_filters, filter_size, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)
            else:
                super(Deconv2DLayer, self).__init__(incoming, num_filters, filter_size, W=W, **kwargs)
            # add the binary tag to weights
            self.params[self.W]=set(['binary'])
        else:
            if W is None:
                super(Deconv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)
            else:
                super(Deconv2DLayer, self).__init__(incoming, num_filters, filter_size, W=W, **kwargs)

    # see BaseConvLayer
    def convolve(self, input, deterministic=False, **kwargs):

        self.Wb = binarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        #
        #self.Wb = T.printing.Print('Wb:')(self.Wb)
        #self.Wb_shape = T.printing.Print('Wb_shape')(self.Wb.shape)
        #print 'Wb.shape:', type(self.Wb.shape), self.Wb.shape.eval()
        Wr = self.W
        self.W = self.Wb

        rvalue = super(Deconv2DLayer, self).convolve(input, **kwargs)

        self.W = Wr

        return rvalue

def scan_in_sample(x_i, filts):
    #print '@scan_in_sample'
    assert x_i.ndim==3
    assert filts.ndim==4

    # fn 個のフィルターを適用 : fn 個の出力
    results, updates = theano.scan(
        fn = perform_filters,
        sequences = filts,
        non_sequences=x_i,
        outputs_info=None,
    )
    return results

def perform_filters(
        #seq
        filter_j,
        #prior
        #non-seq
        x_i
):
    #print '@perform_filters'
    assert filter_j.ndim==3
    assert x_i.ndim==3

    # ch 方向に和を取る
    return T.tensordot(x_i.flatten(ndim=2), filter_j,axes=2)

    ### Equivalent Code:>>
    # sum0 = T.zeros_like( T.dot( x_i[0].flatten(), filter_j[0] ) )
    # results, updates = theano.scan(
    #     fn = accum_filter,
    #     sequences = [x_i.flatten(ndim=2), filter_j],
    #     outputs_info = sum0
    # )
    # #results = theano.printing.Print("perform_filters.results:")(results)
    # print '@perform_filters.3'
    # return results[-1]
    ### Equivalent Code:<<

# by H.Terada
class DeconvSparse2DLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_filters, filter_size, output_size,
                 binary=True,
                 flip_filters=False,
                 W=None,
                 stride=1,
                 crop=0, **kwargs):
        super(DeconvSparse2DLayer,self).__init__(incoming, **kwargs)

        self.binary=binary
        self.flip_filters=flip_filters

        if W is None:
            self.W = lasagne.init.Normal().sample( (num_filters, incoming.output_shape[0], filter_size, filter_size) )
        else:
            # To fit the lasagne TransposedConv2D's spec, make logic of flip inverted
            if self.flip_filters:
                self.W = W
            else:
                # flipping
                self.W = W[:, :, ::-1, ::-1]

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.output_size = output_size
        self.stride = stride
        self.crop = crop

        assert incoming.output_shape[2] == incoming.output_shape[3]
        C, Z = self.make_C(self.num_filters, incoming.output_shape[1], incoming.output_shape[2],
                        self.filter_size, self.output_size, self.stride, self.crop)
        #print 'C.shape=', C.shape, type(C)
        self.C = self.add_param(C, C.shape, name='C', trainable=True)
        #self.Z = self.add_param(Z, Z.shape, name='Z')
        self.Z = Z

    def make_C(self, num_filters, num_channels, input_size, filter_size, output_size, stride, crop):
        #print 'make_C: num_filters %d, num_channels %d' % (num_filters, num_channels)
        C_e_shape = (input_size*input_size, output_size*output_size)

        # loop: filter number
        l_filters = []
        l_zeros = []
        for fn in range(0, num_filters):

            # loop: channel
            l_chs = []
            l_zero_chs = []
            for ch in range(0, num_channels):
                l_val=[]
                l_row=[]
                l_col=[]

                l_zero_flag=[]
                l_zero_row=[]
                l_zero_col=[]

                # loop : C row, col
                for row in range(0, C_e_shape[0]):
                    for col in range(0, C_e_shape[1]):
                        w_row = (col/output_size) + crop - (row/input_size) * stride
                        w_col = (col%output_size) + crop - (row%input_size) * stride

                        if (w_row <0 or w_row > filter_size-1 or
                            w_col <0 or w_col > filter_size-1):
                            # zero
                            l_zero_flag.append(1.0)
                            l_zero_row.append(row)
                            l_zero_col.append(col)
                            pass
                        else:
                            # val
                            l_val.append( self.W[fn, ch, w_row, w_col] )

                            l_row.append( row )
                            l_col.append( col )

                # print 'l_row:', l_row
                # print len(l_row), len(l_col)
                C_e = sp.csr_matrix((l_val, (l_row, l_col)), shape=C_e_shape ).toarray()
                l_chs.append( C_e )

                # zero mask
                Z_e = sp.csr_matrix((l_zero_flag, (l_zero_row, l_zero_col)), shape=C_e_shape ).toarray()
                l_zero_chs.append( Z_e )

            l_filters.append( l_chs )
            l_zeros.append( l_zero_chs )

        C = np.array(l_filters, dtype=floatT)
        Z = np.array(l_zeros, dtype=floatT)
        return C, Z


    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_filters, self.output_size, self.output_size)

    def get_output_for(self, input, **kwargs):
        assert input.ndim==4
        out_shape = self.get_output_shape_for(input.shape)

        #print 'out_shape', out_shape
        #print 'input:', type(input), input.ndim

        Cb = binarization2(self.C, self.Z, 1.0) if self.binary else self.C
        deconvs, updates = theano.scan(
            fn = scan_in_sample,
            sequences = input,
            non_sequences = Cb,
            outputs_info=None
        )
        out = deconvs.reshape(out_shape)
        return out

# by Hideo Terada
class BatchNormLayer(lasagne.layers.BatchNormLayer):
    def __init__(self, incoming, H, verbose=False, axes='auto', epsilon=1e-4, alpha=0.1,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1), **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        self.verbose = verbose
        self.H = H

    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None, **kwargs):
        input_mean = input.mean(self.axes)
        input_inv_std = T.inv(T.sqrt(input.var(self.axes) + self.epsilon))

        # Decide whether to use the stored averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        if use_averages:
            mean = self.mean
            inv_std = self.inv_std
        else:
            mean = input_mean
            inv_std = input_inv_std

        # Decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            inv_std += 0 * running_inv_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        inv_std = inv_std.dimshuffle(pattern)

        if False:
            ## normalize
            normalized = (input - mean) * (gamma * inv_std) + beta

            return binary_tanh_unit(normalized)
        else:

            #---- by H.Terada from here
            # FINN style activation by threshold

            gi = gamma*inv_std
            tau = round3(mean - beta/gi)

            if self.verbose:
                print 'gi.shape:', gi.shape
                print 'tau.shape:', tau.shape
                #gi = T.printing.Print('gi:')(gi)
                #tau = T.printing.Print('tau:')(tau)

            # case1 = T.switch( T.ge(tau, input), +1, -1 )
            # case2 = T.switch( T.ge(input, tau), +1, -1 )
            # output = T.switch( T.lt(gi,0),  # if gi < 0
            #                    case1,
            #                    case2
            # )

            #output = T.switch( T.ge( input, tau), +1, -1 ) #NG

            output = binary_tanh_unit2( input - tau, self.H ) #OK

            return output

