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

## Run
~~~
$cd b-dcgan/training
$python bdcgan_mnist_train.py
~~~
### Scenario setting
The parameters of **scenario** can be modified in source code directly:
- in b-dcgan/training/binary_mnist.py
-- The property variables of class BinaryMnist

