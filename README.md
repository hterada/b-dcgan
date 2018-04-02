# B-DCGAN : Binary-Deep Convolutional & Conditional Generative Adversarial Networks
### [Hideo Terada](http://terada-h.hatenablog.com/), [Hayaru Shouno](http://daemon.inf.uec.ac.jp/ja/)
___
This codes are described in the article [B-DCGAN:Evaluation of Binarized DCGAN for FPGA](https://arxiv.org/abs/1803.10930).

## Setup
### Environment
- Liunx PC(cf. Ubuntu) with NVIDIA Cuda GPU (use used TITAN-X)
- python 2.7
- [Theano 1.0](http://deeplearning.net/software/theano/)
- [Lasagne 0.2](https://lasagne.readthedocs.io/en/latest/)

### Prepare the Dataset
- Place the [MNIST datasets](http://yann.lecun.com/exdb/mnist/) into a directory
  - default directory: **~/datasets** ; it is defined in file 'b-dcgan/training/dcgan_code/lib/config.py').
- Each file must be unzipped.
- File List: (filename must be set as below)
  - t10k-images.idx3-ubyte
  - t10k-labels.idx1-ubyte
  - train-images.idx3-ubyte
  - train-labels.idx1-ubyte

## Run Training
~~~
$cd b-dcgan/training
$python bdcgan_mnist_train.py
~~~
### Scenario setting
The parameters of **scenario** can be modified in source code directly:
- in b-dcgan/training/binary_mnist.py
-- The property variables of class BinaryMnist

### Result of Training
- Model parameters are in sub directory 'models'.
- Generated images are in sub directory 'samples'

## Convert Trained Model to C++ header
Because of historical reason, there are 2 kinds of trained model files which concern with the B-DCGAN generator.
One is called 'gen_params' (generator params), The other is called 'gen_sp_params' (generator special params).
- The 'gen_params' includes Weights of FullConnect layers and Deconvolution layers, and also includes 'beta' and 'gamma' of BatchNormalization.
- The 'gen_sp_params' includes 'mean' and 'inv_std' of BatchNormalization.
- Practically, these files are named as below:
  - \<N\>_gen_params.jl
  - \<N\>_gen_params.jl
- \<N\> is the number of training iteration of 4-digits.

To convert these params to C++ header, do like this:
~~~
$cd b-dcgan/training
$python model_to_ch.py 0082_gen_params.jl 0082_gen_sp_params.jl
~~~
**CAUTION:** The \<N\> value of two files must be same value.

The results are:
~~~
$ls *.h
model_deconv_W_0.h
model_deconv_W_1.h
model_dense_W_0.h
model_dense_W_1.h
model_dense_W_2.h
model_dense_W_3.h
model_sp_BinBN_tau_0.h
model_sp_BinBN_tau_1.h
model_sp_BinBN_tau_2.h
model_sp_BinBN_tau_3.h
model_sp_BinBN_tau_4.h
~~~

### Setting of 'model_to_ch.py'
The 'model_to_ch.py' runs by refering python object of class 'BinaryMnist' in 'binary_mnist.py' module which is used in training-time.
So you should let the object's property(flags) 'as-is' when it was used at training.


