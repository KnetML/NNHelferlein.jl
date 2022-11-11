
# Networks and chains

+ [`AbstractNN`](@ref) - *Helferlein* network type
+ [`AbstractChain`](@ref) - *Helferlein* chain type


+ [`Classifier`](@ref) - network with NLL loss
+ [`Regressor`](@ref) - network with MSE soll
+ [`VAE`](@ref) - variational autoencoder wrapper

+ [`Chain`](@ref)

### Network helpers

+ [`add_layer!`](@ref)
+ [`+`](@ref add_layer!)

+ [`summary`](@ref)
+ [`save_network`](@ref) - save as jld2 file
+ [`load_network`](@ref)
+ [`copy_network`](@ref) - copy from and to GPU



# Layers

+ [`AbstractLayer`](@ref)

### Fully connected layers

+ [`Dense`](@ref)
+ [`Linear`](@ref)
+ [`Embed`](@ref)
+ [`FeatureSelection`](@ref)


### Convolutional

Layers for convolutional networks:

+ [`Conv`](@ref)
+ [`DeConv`](@ref)
+ [`ResNetBlock`](@ref)
+ [`DepthwiseConv`](@ref)
+ [`Pool`](@ref)
+ [`UnPool`](@ref)
+ [`Pad`](@ref)
+ [`Flat`](@ref)
+ [`PyFlat`](@ref)
+ [`GlobalAveragePooling`](@ref)

### Recurrent

Layers for recurrent networks:

+ [`Recurrent`](@ref) - type for recurrent layers
+ [`RecurrentUnit`](@ref) - type for recurrent units


#### Helpers for recurrent networks

+ [`get_hidden_states`](@ref)
+ [`get_cell_states`](@ref)
+ [`set_hidden_states!`](@ref)
+ [`set_cell_states`](@ref)!
+ [`reset_hidden_states!`](@ref)
+ [`reset_cell_states!`](@ref)





### Other layers

+ [`Activation`](@ref)
+ [`Softmax`](@ref)
+ [`Logistic`](@ref)
+ [`Dropout`](@ref)
+ [`BatchNorm`](@ref)
+ [`LayerNorm`](@ref)
+ [`GaussianNoise`](@ref)


### Attention Mechanisms

+ [`AttentionMechanism`](@ref)
+ [`AttnBahdanau`](@ref)
+ [`AttnLuong`](@ref)
+ [`AttnDot`](@ref)
+ [`AttnLocation`](@ref)
+ [`AttnInFeed`](@ref)



### Layers and helpers for transformers

+ [`PositionalEncoding`](@ref)

+ [`mk_padding_mask`](@ref)
+ [`mk_peek_ahead_mask`](@ref)

+ [`dot_prod_attn`](@ref)
+ [`MultiHeadAttn`](@ref)
+ [`separate_heads`](@ref)
+ [`merge_heads`](@ref)



# Data provider utilities

+ [`DataLoader`](@ref) - type for iterator of minibatches

### For tabular data

+ [`dataframe_read`](@ref)
+ [`dataframe_minibatch`](@ref) - turn a dataframe into minibatches
+ [`dataframe_split`](@ref)
+ [`mk_class_ids`](@ref)


### For image data

+ [`ImageLoader`](@ref)

+ [`mk_image_minibatch`](@ref)
+ [`get_class_labels`](@ref)

#### Image to array tools

+ [`image2array`](@ref)
+ [`array2image`](@ref)
+ [`array2RGB`](@ref)

#### ImageNet tools

+ [`preproc_imagenet`](@ref)
+ [`predict_imagenet`](@ref)
+ [`get_imagenet_classes`](@ref)


### Text data

+ [`WordTokenizer`](@ref)

+ [`sequence_minibatch`](@ref)
+ [`pad_sequence`](@ref)
+ [`truncate_sequence`](@ref)

#### Text corpus example data download

+ [`get_tatoeba_corpus`](@ref)



# Iteration utilities

+ [`PartialIterator`](@ref)
+ [`split_minibatches`](@ref)
+ [`MBNoiser`](@ref)
+ [`MBMasquerade`](@ref)




# Training

+ [`tb_train!

### Evaluation and accuracy

+ [`predict`](@ref)
+ [`predict_top5`](@ref)
+ [`minibatch_eval`](@ref)
+ [`confusion_matrix`](@ref)

### Loss functions

+ [`focal_nll`](@ref)
+ [`focal_bce`](@ref)

### Accuracy functions

+ [`squared_error_acc`](@ref)
+ [`abs_error_acc`](@ref)
+ [`hamming_dist`](@ref)
+ [`peak_finder_acc`](@ref)






# Other utils
### Utils for array manipulation

+ [`crop_array`](@ref)
+ [`blowup_array`](@ref)
+ [`recycle_array`](@ref)
+ [`de_embed`](@ref)

### Utils for fixing types in GPU context

+ [`init0`](@ref)
+ [`convert2CuArray`](@ref)
+ [`ifgpu`](@ref)
+ [`emptyCuArray`](@ref)


### Datasets

+ [`dataset_mit_nsr`](@ref)
+ [`dataset_mnist`](@ref)
+ [`dataset_iris`](@ref)
+ [`get_tatoeba_corpus`](@ref)

# Pretrained networks

+ [`get_vgg16`](@ref)
+ [`get_resnet50v2`](@ref)