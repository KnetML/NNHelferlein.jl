API doc of all exported functions are listed here:

# Chains

```@docs
AbstractNN
add_layer!
+
print_network
```

```@docs
Classifier
Regressor
Chain
VAE
```

# Layers

```@docs
Layer
```

## Fully connected layers

```@docs
Dense
Linear
Embed
```

## Convolutional

```@docs
Conv
DeConv
ResNetBlock
DepthwiseConv
Pool
UnPool
Pad
```

## Recurrent

```@docs
RecurrentUnit
Recurrent
get_hidden_states
get_cell_states
set_hidden_states!
set_cell_states!
reset_hidden_states!
reset_cell_states!
```

## Others

```@docs
Flat
PyFlat
FeatureSelection
Activation
Softmax
Logistic
Dropout
BatchNorm
LayerNorm
GaussianNoise
GlobalAveragePooling
```


## Attention Mechanisms

```@docs
AttentionMechanism
AttnBahdanau
AttnLuong
AttnDot
AttnLocation
AttnInFeed
```



# Data providers

```@docs
DataLoader
```

## Iteration utilities
```@docs
PartialIterator
split_minibatches
MBNoiser
MBMasquerade
```

## Tabular data

Tabular data is normally provided in table form (csv, ods)
row-wise, i.e. one sample per row.
The helper functions can read the tables and generate Knet compatible
iterators of minibatches.

```@docs
dataframe_read
dataframe_minibatch
dataframe_split
mk_class_ids
```

## Image data

Images as data should be provided in directories with the directory names
denoting the class labels.
The helpers read from the root of a directory tree in which the
first level of sub-dirs tell the class label. All images in the
tree under a class label are read as instances of the respective class.
The following tree will generate the classes `daisy`, `rose` and `tulip`:

```
image_dir/
├── daisy
│   ├── 01
│   │   ├── 01
│   │   ├── 02
│   │   └── 03
│   ├── 02
│   │   ├── 01
│   │   └── 02
│   └── others
├── rose
│   ├── big
│   └── small
└── tulip
```

```@docs
ImageLoader
mk_image_minibatch
get_class_labels
image2array
array2image
array2RGB
```

## Text data

```@docs
WordTokenizer
get_tatoeba_corpus
sequence_minibatch
pad_sequence
truncate_sequence
```



# Training

```@docs
tb_train!
```

# Evaluation and accuracy

```@docs
focal_nll
focal_bce
predict
predict_top5
minibatch_eval
squared_error_acc
abs_error_acc
hamming_dist
peak_finder_acc
confusion_matrix
```

# ImageNet tools

```@docs
preproc_imagenet
predict_imagenet
get_imagenet_classes
```


# Other utils

## Layers and helpers for transformers

```@docs
PositionalEncoding
mk_padding_mask
mk_peek_ahead_mask
dot_prod_attn
MultiHeadAttn
separate_heads
merge_heads
```


## Utils for array manipulation

```@docs
crop_array
blowup_array
recycle_array
de_embed
```

## Utils for fixing types in GPU context

```@docs
init0
convert2CuArray
convert2KnetArray
ifgpu
emptyCuArray
```

## Saving, loading and inspection of models

```@docs
save_network
load_network
copy_network
summary
print_network
```

## Datasets

```@docs
dataset_mit_nsr
dataset_mnist
dataset_iris
```

# Pretrained networks

```@docs
get_vgg16
get_resnet50v2
```