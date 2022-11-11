
# Networks and chains

+ [`AbstractNN`](@ref) - *Helferlein* network type
+ [`AbstractChain`](@ref) - *Helferlein* chain type


+ Classifier
+ Regressor
+ Chain
+ VAE

### Network helpers

add_layer!
+
summary

save_network
load_network
copy_network



# Layers

AbstractLayer

## Fully connected layers

Dense
Linear
Embed
FeatureSelection


## Convolutional

Conv
DeConv
ResNetBlock
DepthwiseConv
Pool
UnPool
Pad
Flat
PyFlat
GlobalAveragePooling

## Recurrent

Recurrent


### Helpers for recurrent networks

RecurrentUnit
get_hidden_states
get_cell_states
set_hidden_states!
set_cell_states!
reset_hidden_states!
reset_cell_states!





## Other layers

Activation
Softmax
Logistic
Dropout
BatchNorm
LayerNorm
GaussianNoise


## Attention Mechanisms

AttentionMechanism
AttnBahdanau
AttnLuong
AttnDot
AttnLocation
AttnInFeed



## Layers and helpers for transformers

PositionalEncoding

mk_padding_mask
mk_peek_ahead_mask

dot_prod_attn
MultiHeadAttn
separate_heads
merge_heads



# Data provider utilities

DataLoader

## For tabular data

dataframe_read
dataframe_minibatch
dataframe_split
mk_class_ids


## For image data

ImageLoader

mk_image_minibatch
get_class_labels

### Image to array tools

image2array
array2image
array2RGB

### ImageNet tools

preproc_imagenet
predict_imagenet
get_imagenet_classes


## Text data

WordTokenizer

sequence_minibatch
pad_sequence
truncate_sequence

### Text corpus example data download

get_tatoeba_corpus



# Iteration utilities

PartialIterator
split_minibatches
MBNoiser
MBMasquerade




# Training

tb_train!

## Evaluation and accuracy

predict
predict_top5
minibatch_eval
confusion_matrix

### Loss functions

focal_nll
focal_bce

### Accuracy functions

squared_error_acc
abs_error_acc
hamming_dist
peak_finder_acc






# Other utils
## Utils for array manipulation

crop_array
blowup_array
recycle_array
de_embed

## Utils for fixing types in GPU context

init0
convert2CuArray
ifgpu
emptyCuArray


## Datasets

dataset_mit_nsr
dataset_mnist
dataset_iris
get_tatoeba_corpus

# Pretrained networks

get_vgg16
get_resnet50v2