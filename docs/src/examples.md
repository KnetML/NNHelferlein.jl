# Examples

Examples may be used as templates for new projects...    
All examples are at [GitHub/examples](https://github.com/KnetML/NNHelferlein.jl/tree/main/examples):

+ [`Simple MLP`]
  (https://github.com/KnetML/NNHelferlein.jl/blob/main/examples/01-simple-mlp.ipynb):
  A simple multi-layer perceptron for MNIST classification,
  build with Knet and *Helferlein*-types in just one line of code (or so).


+ [`Simple LeNet`]
  (https://github.com/KnetML/NNHelferlein.jl/blob/main/examples/10-simple-lenet.ipynb):
  A simple LeNet for MNIST classification, 
  build with help of the *Helferlein* layers in just two (ok: long) lines of code. 

+ [`Training unbalanced data with help of a focal loss function`]
  (https://github.com/KnetML/NNHelferlein.jl/blob/main/examples/11-focal_loss.ipynb):
  A simple MLP with focal loss demonstrate classification of highly unbalanced data.


+ [`Vanilla Autoencoder`]
  (https://github.com/KnetML/NNHelferlein.jl/blob/main/examples/30-ae.ipynb):
  A simple autoencoder design with help of *Knet* in *Helferlein*-style.
  

+ [`Convolutional Autoencoder`]
  (https://github.com/KnetML/NNHelferlein.jl/blob/main/examples/31-cae.ipynb):
  A convolutional autoencoder design with help of *Knet* in *Helferlein*-style.
  

+ [`Variational Autoencoder`]
  (https://github.com/KnetML/NNHelferlein.jl/blob/main/examples/32-vae.ipynb):
  Example for a simple VAE utilising the NNHelferlein-type `VAE` and demonstrating the
  fascinating regularisation of a VAE.

+ [`Simple sequence-to-sequence network`]
  (https://github.com/KnetML/NNHelferlein.jl/blob/main/examples/60-s2s-nlp-gru.ipynb):
  Simple s2s network to demonstrate how to setup macghine translation with 
  a rnn.

+ [`Sequence-to-sequence RNN for machine translation`]
  (https://github.com/KnetML/NNHelferlein.jl/blob/main/examples/61-RNN_bi_attn.ipynb):
  RNN to demonstrate how to setup machine translation with 
  a bidirectional encoder RNN and attention.

+ [`RNN Sequence tagger for annotation of ECGs`]
  (https://github.com/KnetML/NNHelferlein.jl/blob/main/examples/62-ECG-tagger.ipynb):
  RNN to demonstrate how to set-up a sequence tagger to detect
  heart beats. Only one layer with 8 units is necessary to achieve almost
  100% correct predictions. 
  The example includes the definition on peephole LSTMs to display
  how to integrate non-standard rnn-units with the *NNHelfrelein* framework.

+ [`Import a Keras model`]
  (https://github.com/KnetML/NNHelferlein.jl/blob/main/examples/75-keras-model.ipynb):
  The notebook shows the import of a pretrained VGG16 model
  from Tensorflow/Keras into a Knet-style CNN
  and its application to example images utilising the
  *Helferlein* imagenet-utilities.


+ [`Transformer for machine translation`]
  (https://github.com/KnetML/NNHelferlein.jl/blob/main/examples/80-transformer.ipynb):
  A simple transformer architecture is set up according to the
  2017 Vaswani paper *Attention is All You Need* with help of 
  *NNHelferlein*-utils.


+ [`Simple Transformer API for Bert-like architectures`]
  (https://github.com/KnetML/NNHelferlein.jl/blob/main/examples/81-transformer-api.ipynb):
  A simple transformer architecture is set up with the *NNHelferlein* transformer API.



### Pretrained Nets
Based on the Keras import constructors, it is easy to 
import  pretrained models from the TF/Keras ecosystem.
+ [`VGG16`]
  (https://github.com/KnetML/NNHelferlein.jl/blob/main/examples/70-pretrained_vgg16.ipynb)

+ [`ResNet50 V2`]
  (https://github.com/KnetML/NNHelferlein.jl/blob/main/examples/71-pretrained_resnet50v2.ipynb)