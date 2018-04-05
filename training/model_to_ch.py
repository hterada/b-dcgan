# -*- coding:utf-8 -*-

import sys
sys.path.append('dcgan_code')
sys.path.append('dcgan_code/lib')
sys.path.append('BinaryNet/Train-time')

import os
import re
from time import time
import numpy as np
import math
from sklearn.externals import joblib

import binary_net
from binary_net import round3
from binary_mnist import BinaryMnist

def val_func_binary(val):
    if val >= 0.0:
        return '1'
    else:
        return '0'

def val_func_boolean(val):
    if val >= 0.0:
        return 'true '
    else:
        return 'false'

def val_func_float(val):
    return '%g' % val

def val_func_int(val):
    return '%d' % int(np.round(val))

def val_func_ap2(val):
    return '%d' % int(np.round(math.log(val,2)))

CHUNK_SIZE=32

def write_param_bin(out_f, nda, name, suffix=[]):
    indent = ' '*len(suffix)
    out_f.write('%s{\n' % indent)
    if nda.ndim==1:
        at_first=True
        count=0
        for ix in range(0, len(nda), CHUNK_SIZE):
            out_val=int(0)
            for bit in range(0, CHUNK_SIZE):
                if ix+bit > len(nda)-1:
                    break

                # set 1 if input value > 0.0 else 0
                if nda[ix+bit] > 0.0:
                    out_val |= (1<<bit)

            if at_first:
                out_f.write( '0x%08x' % out_val )
                at_first=False
            else:
                out_f.write( ',%s0x%08x' % ('\n%s'%indent if count%8==0 else ' ', out_val) )

            count += 1

        out_f.write('}')
    else:
        at_first=True
        ix=0
        for suba in nda:
            suffix.append(ix)
            if at_first:
                out_f.write('\n%s//%s%s\n' % (indent, name, str(suffix)) )
                at_first=False
            else:
                out_f.write(',\n%s//%s%s\n' % (indent, name, str(suffix)) )

            write_param(out_f, suba, val_func, name, suffix)
            ix += 1
            suffix.pop()

        out_f.write('\n%s}' % indent)
    return

def write_param(out_f, nda, name, val_func, suffix=[]):
    indent = ' '*len(suffix)
    out_f.write('%s{\n' % indent)
    if nda.ndim==1:
        at_first=True
        count=0
        for val in nda:
            if at_first:
                out_f.write( '%s' % val_func(val) )
                at_first=False
            else:
                out_f.write( ',%s%s' % ('\n%s'%indent if count%8==0 else ' ', val_func(val) ) )

            count += 1

        out_f.write('}')
    else:
        at_first=True
        ix=0
        for suba in nda:
            suffix.append(ix)
            if at_first:
                out_f.write('\n%s//%s%s\n' % (indent, name, str(suffix)) )
                at_first=False
            else:
                out_f.write(',\n%s//%s%s\n' % (indent, name, str(suffix)) )

            write_param(out_f, suba, val_func, name, suffix)
            ix += 1
            suffix.pop()

        out_f.write('\n%s}' % indent)
    return

def write_shape(param, name, out_file):
    out_file.write('\nstatic const int %s_shape[] = {' % name )
    at_first=True
    for sh in param.shape:
        if at_first:
            out_file.write('%d' % sh)
            at_first=False
        else:
            out_file.write(',%d' %sh)
    out_file.write('};\n')

def write_array_as_binary(param, name, out_file, flip_filters=False):
    #
    # output array original shape
    #
    write_shape(param, name, out_file)

    #
    # output array
    #
    out_param=None
    if flip_filters:
        # flip fliters
        out_param = param[:, :, ::-1, ::-1]
        out_param = out_param.flatten()
    else:
        out_param = param.flatten()

    dim_str = '[%d]' % ( (out_param.shape[0]+(CHUNK_SIZE-1)) / CHUNK_SIZE )

    #out_file.write('static const %s %s%s = ' % ('ap_uint<%d>'%CHUNK_SIZE , name, dim_str))
    out_file.write('static const %s %s%s = ' % ('uint32', name, dim_str))
    write_param_bin(out_file, out_param, name)
    out_file.write(';\n')

    return

def write_array_as_real(param, typename, name, comment, out_file, flip_filters):
    #
    # output array original shape
    #
    write_shape(param, name, out_file)

    #
    # output array
    #
    out_param = None
    if flip_filters:
        out_param = param[:, :, ::-1, ::-1]
        out_param = out_param.flatten()
    else:
        out_param = param.flatten()

    dim_str=''
    for sh in out_param.shape:
        dim_str += '[%d]' % sh

    out_file.write('// %s\n' % comment )
    out_file.write('static const %s %s%s = ' % (typename, name, dim_str))
    write_param(out_file, out_param.flatten(), name, val_func_float)

    out_file.write(';\n')

    return

def open_out_file(frm, filename):
    out_file = open(filename , 'w')
    out_file.write('#pragma once\n')
    out_file.write('// %s\n' % filename )
    out_file.write('// Binady-DCGAN params from %s\n' % frm )

    return out_file

def printParamProfile(tvalue):
    #print param.shape
    #print param
    param = tvalue.get_value()
    print 'name:%s, shape:%s, mean:%g, std:%g, min:%g, max:%g' % ( tvalue.name, param.shape, param.mean(), param.std(), param.min(), param.max() )

def convert_gen_params( input_filename, binaryMnist ):
    print '**** convert_gen_params ****'
    list_tvalue = joblib.load( input_filename )
    print 'len(list_tvalue):', len(list_tvalue)


    out_bin_beta = []
    out_bin_gamma = []
    out_beta = []
    out_gamma = []
    cnt_W=0
    cnt_beta=0
    cnt_gamma=0
    cnt_mean=0

    NUM_ENCODER_LAYER = binaryMnist.NUM_HIDDEN_LAYERS + 1

    for tvalue in list_tvalue:
        # param name
        printParamProfile(tvalue)

        if re.match('.*W', tvalue.name):
            if cnt_W < NUM_ENCODER_LAYER:
                # Encoder
                # for Dense(FC) Layer
                filename = 'model_dense_W_%d.h' % cnt_W
                print 'write file:', filename
                with open_out_file(input_filename, filename) as out_file:
                    assert tvalue.ndim==2
                    if binaryMnist.IS_USE_B_FC==True:
                        # bin Dense W
                        write_array_as_binary(tvalue.get_value(), 'param_bin_dense_W%d' % cnt_W , out_file, flip_filters=False)
                    else:
                        # real Dense W
                        write_array_as_real(tvalue.get_value(), 'DenseWType', 'param_dense_W%d' % cnt_W, 'Dense param', out_file, flip_filters=False)

            else:
                # Decoder
                # deconv W
                no = cnt_W - NUM_ENCODER_LAYER
                filename = 'model_deconv_W_%d.h' % no
                print 'write file:', filename
                with open_out_file(input_filename, filename) as out_file:
                    assert tvalue.ndim==4
                    if no == 0:
                        # Deconv-1
                        if binaryMnist.IS_USE_B_DECONV_1==True:
                            write_array_as_binary(tvalue.get_value(), 'param_bin_deconv_W%d' % no, out_file, flip_filters=True)
                        else:
                            write_array_as_real(tvalue.get_value(), 'DeconvWType', 'param_deconv_W%d' % no , 'Deconv param', out_file, flip_filters=True)
                    else:
                        # Deconv-2
                        if binaryMnist.IS_USE_B_DECONV_2==True:
                            write_array_as_binary(tvalue.get_value(), 'param_bin_deconv_W%d' % no, out_file, flip_filters=True)
                        else:
                            write_array_as_real(tvalue.get_value(), 'DeconvWType', 'param_deconv_W%d' % no , 'Deconv param', out_file, flip_filters=True)

            # to next
            cnt_W += 1

        elif re.match('.*beta', tvalue.name):
            assert tvalue.ndim==1
            if cnt_beta < NUM_ENCODER_LAYER:
                # Encoder
                if binaryMnist.IS_USE_B_BNA_1==True:
                    # bin BN beta
                    out_bin_beta.append( tvalue )
                else:
                    # real BN beta
                    out_beta.append( tvalue )
            else:
                # Decoder
                if binaryMnist.IS_USE_B_BNA_2==True:
                    # bin BN beta
                    out_bin_beta.append( tvalue )
                else:
                    # real BN beta
                    out_beta.append( tvalue )
            cnt_beta += 1
        elif re.match('.*gamma', tvalue.name):
            if cnt_gamma < NUM_ENCODER_LAYER:
                # Encoder
                if binaryMnist.IS_USE_B_BNA_1==True:
                    # bin BN gamma
                    out_bin_gamma.append( tvalue )
                else:
                    # real BN beta
                    out_gamma.append( tvalue )
            else:
                # Decoder
                if binaryMnist.IS_USE_B_BNA_2==True:
                    # bin BN gamma
                    out_bin_gamma.append( tvalue )
                else:
                    # real BN gamma
                    out_gamma.append( tvalue )
            cnt_gamma += 1
        elif re.match('.*\.b$', tvalue.name):
            assert tvalue.ndim==1
            # b, skip
            assert tvalue.get_value().min() == 0
            assert tvalue.get_value().max() == 0

    return out_bin_beta, out_bin_gamma, out_beta, out_gamma

def convert_special_params( input_filename, binaryMnist, list_bin_beta, list_bin_gamma, list_beta, list_gamma ):
    l_params = joblib.load( input_filename )
    print 'len(l_params)', len(l_params)

    param_no=0
    list_bin_mean = []
    list_bin_inv_std = []
    list_mean = []
    list_inv_std = []
    # 一旦パラメータを種類別に振り分ける
    cnt_mean=0
    cnt_inv_std=0
    NUM_ENCODER_LAYER = binaryMnist.NUM_HIDDEN_LAYERS + 1
    for tvalue in l_params:
        printParamProfile(tvalue)

        if re.match('.*mean', tvalue.name):
            if cnt_mean < NUM_ENCODER_LAYER:
                # Encoder
                if binaryMnist.IS_USE_B_BNA_1==True:
                    # bin BN
                    list_bin_mean.append( tvalue )
                else:
                    # real BN
                    list_mean.append( tvalue )
            else:
                # Decoder
                if binaryMnist.IS_USE_B_BNA_2==True:
                    # bin BN
                    list_bin_mean.append( tvalue )
                else:
                    # real BN
                    list_mean.append( tvalue )

            cnt_mean += 1

        elif re.match('.*inv_std', tvalue.name):
            if cnt_inv_std < NUM_ENCODER_LAYER:
                # Encoder
                if binaryMnist.IS_USE_B_BNA_1==True:
                    # bin BN
                    list_bin_inv_std.append( tvalue )
                else:
                    # real BN
                    list_inv_std.append( tvalue )
            else:
                # Decoder
                if binaryMnist.IS_USE_B_BNA_2==True:
                    # bin BN
                    list_bin_inv_std.append( tvalue )
                else:
                    # real BN
                    list_inv_std.append( tvalue )

            cnt_inv_std += 1

    print 'len(list_bin_beta), len(list_bin_gamma):', len(list_bin_beta), len(list_bin_gamma)
    print 'len(list_beta), len(list_gamma):', len(list_beta), len(list_gamma)
    print 'len(list_mean):', len(list_mean)

    assert len(list_bin_beta) == len(list_bin_gamma)
    assert len(list_beta) == len(list_gamma)
    assert len(list_mean) == len(list_inv_std)

    # Bin BN
    for ix in range(0, len(list_bin_beta)):
        filename = 'model_sp_BinBN_tau_%d.h' % (ix)
        print 'write file:', filename
        with open_out_file(input_filename, filename) as out_file:

            gi = list_bin_gamma[ix] * list_bin_inv_std[ix]
            tau = round3(list_bin_mean[ix] - list_bin_beta[ix]/gi)

            print 'tau:', 'max:', tau.eval().max(), 'min:', tau.eval().min()

            name = 'tau_%d' % ix
            write_array_as_real(tau.eval(), 'TauType', name, 'Bin-BatchNorm tau', out_file, flip_filters=False)

    # Real BN
    for ix in range(0, len(list_beta)):
        filename = 'model_sp_BN_%d.h' % (ix)
        print 'write file:', filename
        with open_out_file(input_filename, filename) as out_file:

            write_array_as_real(list_beta[ix].get_value(), 'BetaType', 'beta_%d' % ix, 'BatchNorm beta', out_file, flip_filters=False)
            write_array_as_real(list_gamma[ix].get_value(), 'GammaType', 'gamma_%d' % ix, 'BatchNorm gamma', out_file, flip_filters=False)
            write_array_as_real(list_mean[ix].get_value(), 'MeanType', 'mean_%d' % ix, 'BatchNorm mean', out_file, flip_filters=False)
            write_array_as_real(list_inv_std[ix].get_value(), 'InvStdType', 'inv_std_%d' % ix, 'BatchNorm inv_std', out_file, flip_filters=False)
#
# main
#
if __name__ == '__main__':
    # to refer scenario flags
    binaryMnist = BinaryMnist(is_binary=True)
    list_bin_beta, list_bin_gamma, list_beta, list_gamma = convert_gen_params( sys.argv[1], binaryMnist )
    convert_special_params( sys.argv[2], binaryMnist, list_bin_beta, list_bin_gamma, list_beta, list_gamma )
