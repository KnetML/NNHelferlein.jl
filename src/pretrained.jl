# API for tensorflow pre-rained networks
#

const PRETRAINED_DIR = joinpath(NNHelferlein.DATA_DIR, "pretrained")

# download pretrained networks from zenodo:
#
# https://github.com/keras-team/keras/blob/v2.9.0/keras/applications/vgg16.py#L43
# https://storage.googleapis.com/tensorflow/
# https://github.com/keras-team/keras/blob/v2.9.0/keras/applications/resnet.py
# https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels.h5
# const VGG19_KERAS_URL = "https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5"

# const VGG16_NAME = "VGG16 (pretrained, TF/Keras)"
# const VGG16_KERAS_URL = "https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
# const VGG16_FILE_NAME = "vgg16.h5"


#const RESNET50V2_NAME = "Resnet50 v2 (pretrained, TF/Keras)"
#const RESNETV2_KERAS_URL = "https://storage.googleapis.com/tensorflow/keras-applications/inception_resnet_v2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5"
#const RESNET50V2_KERAS_URL = "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels.h5"
#const RESNET50V2_FILE_NAME = "resnet50v2.h5"

# Paths to Zenodo files:
#
const ZENODO_DATA_PRETRAINED = "7266776"   # Zenodo identifier
const ZENODO_URL_PRETRAINED = "$ZENODO_URL/record/$ZENODO_DATA_PRETRAINED/files"

const VGG16_NAME = "VGG16 (pretrained, TF/Keras)"
const VGG16_FILE_NAME = "vgg16keras_model.h5"

const VGG19_NAME = "VGG19 (pretrained, TF/Keras)"
const VGG19_FILE_NAME = "vgg19keras_model.h5"

const RESNET50V2_NAME = "Resnet50 v2 (pretrained, TF/Keras)"
const RESNET50V2_FILE_NAME = "resnret50v2keras_model.h5"

const MOBILENETS_NAME = "MobileNet v3 small (pretrained, TF/Keras)"
const MOBILENETS_FILE_NAME = "mobilenetv3smallkeras_model.h5"





function download_pretrained(name, file_name)

    url = "$ZENODO_URL_PRETRAINED/$file_name"
    local_file = "$PRETRAINED_DIR/$file_name"

    if isfile(local_file)
        println("Using already downloaded weights for $name")
    else
        println("Downloading weights for $name from Zenodo")
        println("$url"); flush(stdout)

        if !isdir(PRETRAINED_DIR)
            mkpath(PRETRAINED_DIR)
        end

        Downloads.download(url, local_file)
    end
    return local_file
end

function download_pretrained(url, name, file_name)

    if isfile(file_name)
        println("Using already downloaded weights for $name")
    else
        println("Downloading weights for $name from $url"); flush(stdout)

        if !isdir(PRETRAINED_DIR)
            mkpath(PRETRAINED_DIR)
        end

        Downloads.download(url, file_name)
    end
end



""" 
    function get_vgg16(; filters_only=false, trainable=true)

Return a VGG16 model with pretrained parameters
from Tensorflow/Keras applications API. For details about original model 
and training see
[`Keras Applications`](https://keras.io/api/applications/).

### Arguments
+ `filters_only=false`: if `true`, only the filterstack is returned   
            (without Flatten() and classifier) to be integrated in to 
            any chain.
+ `trainable=true`: if `true`, the filterstack is set trainable, otherwise
            only the classifier part is trainable and the filter weights are 
            fixed.

### Details:
The model weights are imported from the respective Keras *Application*,
which is trained with preprocessed images of size 224x224 pixel.
Image data format must be colour channels `BGR` and 
colour values `0.0 - 1.0`.


This can be re-built by using a preprocessing pipeline and the
*Helferlein*-function `preproc_imagenet_vgg()` from a directory
`img_path` with images:

```julia
pipl = CropRatio(ratio=1.0) |> Resize(224,224)
mini_batches = mk_image_minibatch(img_path, 2, train=false, 
        aug_pipl=pipl, pre_proc=preproc_imagenet_vgg)
```


Model structure is:
[`VGG16 topology plot created by netron`]
(https://github.com/KnetML/NNHelferlein.jl/blob/main/docs/src/assets/netron-vgg16-w200.png)
"""
function get_vgg16(; filters_only=false, trainable=true)

    local_file = joinpath(PRETRAINED_DIR, VGG16_FILE_NAME)
    local_file = download_pretrained(VGG16_NAME, VGG16_FILE_NAME)
    h5 = HDF5.h5open(local_file)

    filter_layers = Chain(
            Conv(h5, "block1_conv1", trainable=trainable, padding=1),
            Conv(h5, "block1_conv2", trainable=trainable, padding=1),
            Pool(),
            Conv(h5, "block2_conv1", trainable=trainable, padding=1),
            Conv(h5, "block2_conv2", trainable=trainable, padding=1),
            Pool(),
            Conv(h5, "block3_conv1", trainable=trainable, padding=1),
            Conv(h5, "block3_conv2", trainable=trainable, padding=1),
            Conv(h5, "block3_conv3", trainable=trainable, padding=1),
            Pool(),
            Conv(h5, "block4_conv1", trainable=trainable, padding=1),
            Conv(h5, "block4_conv2", trainable=trainable, padding=1),
            Conv(h5, "block4_conv3", trainable=trainable, padding=1),
            Pool(),
            Conv(h5, "block5_conv1", trainable=trainable, padding=1),
            Conv(h5, "block5_conv2", trainable=trainable, padding=1),
            Conv(h5, "block5_conv3", trainable=trainable, padding=1),
            Pool())

     classif_layers = Chain(
            Dense(h5, "fc1", trainable=trainable, actf=relu),
            Dense(h5, "fc2", trainable=trainable, actf=relu),
            Dense(h5, "predictions", trainable=trainable, actf=identity)) 

    if filters_only
         vgg = Chain(filter_layers)
    else
        vgg = Classifier( 
            filter_layers,
            PyFlat(python=true),
            classif_layers)
    end

    println("")
    println("Imported pretrained network $VGG16_NAME")
    print_network(vgg)
    return vgg
end





""" 
    function get_resnet50v2(; filters_only=false, trainable=true)

Return a ResNet50 v2 model with pretrained parameters
from Tensorflow/Keras applications API. For details about original model 
and training see
[`Keras Applications`](https://keras.io/api/applications/).

### Arguments
+ `filters_only=false`: if `true`, only the filterstack is returned   
            (without Flatten() and classifier) to be integrated in to 
            any chain.
+ `trainable=true`: if `true`, the filterstack is set trainable, otherwise
            only the classifier part is trainable and the filter weights are 
            fixed.

### Details:
The model weights are imported from the respective Keras *Application*,
which is trained with images of size 224x224 pixel.     
*Cave:* The training set images have not been preprocessed with the 
imagenet default procedure!
In contrats image data format must be colour channels `RGB` and 
colour values `0.0 - 1.0`.

This can be re-built by using a preprocessing pipeline without 
application `preproc_imagenet_resnetv2()` from a directory
`img_path` with images:

```julia
pipl = CropRatio(ratio=1.0) |> Resize(224,224)
mini_batches = mk_image_minibatch(img_path, 2, train=false, 
        aug_pipl=pipl, pre_proc=preproc_imagenet_resnetv2)
```

Model structure is:
[`ResNet50 V2 topology plot created by netron`]
(https://github.com/KnetML/NNHelferlein.jl/blob/main/docs/src/assets/netron-resnet50v2.png)

"""
function get_resnet50v2(; filters_only=false, trainable=true)

    local_file = download_pretrained(RESNET50V2_NAME, RESNET50V2_FILE_NAME)
    h5 = HDF5.h5open(local_file)

    filter_layers = Chain(
        Pad(3),
        Conv(h5, "conv1_conv", trainable=trainable, stride=2, actf=identity),
        Pad(1),
        Pool(;window=3, stride=2),
        BatchNorm(h5, "conv2_block1_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
        Relu(),
        ResNetBlock([
                    Conv(h5, "conv2_block1_1_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv2_block1_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Pad(1),
                    Conv(h5, "conv2_block1_2_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv2_block1_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv2_block1_3_conv", trainable=trainable, actf=identity),
                    ];
                shortcut=[
                    Conv(h5, "conv2_block1_0_conv", trainable=trainable, actf=identity)
                    ]),
        ResNetBlock([
                    BatchNorm(h5, "conv2_block2_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv2_block2_1_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv2_block2_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Pad(1),
                    Conv(h5, "conv2_block2_2_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv2_block2_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv2_block2_3_conv", trainable=trainable, actf=identity),
                    ]), 
        ResNetBlock([
                    BatchNorm(h5, "conv2_block3_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv2_block3_1_conv",use_bias=false,  trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv2_block3_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Pad(1),
                    Conv(h5, "conv2_block3_2_conv", use_bias=false, stride=2, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv2_block3_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv2_block3_3_conv", trainable=trainable, actf=identity),
                    ],
                shortcut = [
                    Pool(;window=1, stride=2)
                    ],
                post = [
                    BatchNorm( h5, "conv3_block1_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu()
                    ]),
        ResNetBlock([
                    Conv(h5, "conv3_block1_1_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm(h5, "conv3_block1_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Pad(1),
                    Conv(h5, "conv3_block1_2_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv3_block1_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block1_3_conv", trainable=trainable, actf=identity),
                    ],
                shortcut = [
                    Conv(h5, "conv3_block1_0_conv", trainable=trainable, actf=identity),
                    ]),  
        ResNetBlock([
                    BatchNorm(h5, "conv3_block2_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block2_1_conv"; use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm(h5, "conv3_block2_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Pad(1),
                    Conv(h5, "conv3_block2_2_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv3_block2_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block2_3_conv", trainable=trainable, actf=identity),
                    ]),
        ResNetBlock([
                    BatchNorm(h5, "conv3_block3_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block3_1_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm(h5, "conv3_block3_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Pad(1),
                    Conv(h5, "conv3_block3_2_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv3_block3_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block3_3_conv", trainable=trainable, actf=identity),
                    ]),
        ResNetBlock([
                    BatchNorm(h5, "conv3_block4_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block4_1_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv3_block4_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Pad(1),
                    Conv(h5, "conv3_block4_2_conv", use_bias=false, trainable=trainable, stride=2, actf=identity),
                    BatchNorm( h5, "conv3_block4_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block4_3_conv", trainable=trainable, actf=identity),
                    ],
                shortcut = [
                    Pool(; window=1, stride=2)
                    ],
                post = [
                    BatchNorm( h5, "conv4_block1_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu()
                    ]),
        ResNetBlock([
                    Conv(h5, "conv4_block1_1_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv4_block1_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Pad(1),
                    Conv(h5, "conv4_block1_2_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv4_block1_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block1_3_conv", trainable=trainable, actf=identity),
                    ];
                shortcut=[
                    Conv(h5, "conv4_block1_0_conv", trainable=trainable, actf=identity)
                    ]),
        ResNetBlock([
                    BatchNorm(h5, "conv4_block2_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block2_1_conv", use_bias=false, trainable=trainable, padding=0, stride=1, actf=identity),
                    BatchNorm(h5, "conv4_block2_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Pad(1),
                    Conv(h5, "conv4_block2_2_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv4_block2_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block2_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ]),
        ResNetBlock([
                    BatchNorm(h5, "conv4_block3_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block3_1_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm(h5, "conv4_block3_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Pad(1),
                    Conv(h5, "conv4_block3_2_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv4_block3_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block3_3_conv", trainable=trainable, actf=identity),
                    ]),
        ResNetBlock([
                    BatchNorm(h5, "conv4_block4_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block4_1_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm(h5, "conv4_block4_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Pad(1),
                    Conv(h5, "conv4_block4_2_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv4_block4_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block4_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ]),
        ResNetBlock([
                    BatchNorm(h5, "conv4_block5_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block5_1_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm(h5, "conv4_block5_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Pad(1),
                    Conv(h5, "conv4_block5_2_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv4_block5_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block5_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ]),
        ResNetBlock([
                    BatchNorm(h5, "conv4_block6_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block6_1_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv4_block6_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Pad(1),
                    Conv(h5, "conv4_block6_2_conv", use_bias=false, trainable=trainable, stride=2, actf=identity),
                    BatchNorm( h5, "conv4_block6_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block6_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ],
                shortcut = [
                    Pool(; window=1, stride=[2,2])
                    ],
                post = [
                    BatchNorm( h5, "conv5_block1_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu()
                    ]),
        ResNetBlock([
                    Conv(h5, "conv5_block1_1_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv5_block1_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Pad(1),
                    Conv(h5, "conv5_block1_2_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv5_block1_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv5_block1_3_conv", trainable=trainable, actf=identity),
                    ];
                shortcut=[
                    Conv(h5, "conv5_block1_0_conv", trainable=trainable, actf=identity)
                    ]),
        ResNetBlock([
                    BatchNorm(h5, "conv5_block2_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv5_block2_1_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm(h5, "conv5_block2_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Pad(1),
                    Conv(h5, "conv5_block2_2_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv5_block2_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv5_block2_3_conv", trainable=trainable, actf=identity),
                    ]),
        ResNetBlock([
                    BatchNorm(h5, "conv5_block3_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv5_block3_1_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm(h5, "conv5_block3_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Pad(1),
                    Conv(h5, "conv5_block3_2_conv", use_bias=false, trainable=trainable, actf=identity),
                    BatchNorm( h5, "conv5_block3_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv5_block3_3_conv", trainable=trainable, actf=identity),
                    ],
                post = [
                    BatchNorm(h5, "post_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu()
                    ])
    )

    classif_layers = Chain(
            Dense(h5, "predictions", trainable=trainable, actf=identity))

    if filters_only
         r50 = Chain(filter_layers)
    else
        r50 = Classifier( 
            filter_layers,
            GlobalAveragePooling(),
            classif_layers
            )
    end

    println("")
    println("Imported pretrained network $RESNET50V2_NAME")
    print_network(r50)
    return r50
end


"""
    struct ResNetBlock <: AbstractChain

Exectable type for one block of a ResNet-type network.

### Constructors:
+ `ResNetBlock(layers; shortcut=[identity], post=[identity])`:
        3 chains to form the block: 
        the main chain, the shortcut and a chain of layers 
        to be added after the confluence.
        All chains must be specified as lists, even if they are 
        empty (`[]`) or comprise only one layer
        (`[BatchNorm]`).
"""
struct ResNetBlock <: AbstractChain
    layers      # [1]: conv layers Chain
                # [2]: Chain in the shortcut route
                # [3]: Chain after rejoining 

    ResNetBlock(layers; shortcut=[identity], post=[identity]) = 
            new([Chain(layers...), Chain(shortcut...), Chain(post...)])

    function (b::ResNetBlock)(x)
        return b.layers[3](b.layers[1](x) .+ b.layers[2](x))
    end
end


function Base.summary(l::ResNetBlock; n=0, indent=0)

    println(" "^indent*"ResNet block with")
    indent += 2

    println(" "^indent*"layers:")
    n += summary(l.layers[1], indent=indent+2)
    
    println(" "^indent*"shortcut:")
    n += summary(l.layers[2], indent=indent+2)
    
    println(" "^indent*"post transformations:")
    n += summary(l.layers[3], indent=indent+2)
    return n
end




# """ 
#     function get_mobilenetv3_small(; filters_only=false, trainable=true)
# 
# Return a ResNet50 v2 model with pretrained parameters
# from Tensorflow/Keras applications API. For details about original model 
# and training see
# [`Keras Applications`](https://keras.io/api/applications/).
# 
# ### Arguments
# + `filters_only=false`: if `true`, only the filterstack is returned   
#             (without Flatten() and classifier) to be integrated in to 
#             any chain.
# + `trainable=true`: if `true`, the filterstack is set trainable, otherwise
#             only the classifier part is trainable and the filter weights are 
#             fixed.
# 
# ### Details:
# The model weights are imported from the respective Keras *Application*,
# which is trained with images of size 224x224 pixel.     
# *Cave:* The training set images have not been preprocessed with the 
# imagenet default procedure!
# 
# This can be re-built by using a preprocessing pipeline without 
# application `preproc_imagenet()` from a directory
# `img_path` with images:
# 
# ```julia
# pipl = CropRatio(ratio=1.0) |> Resize(224,224)
# mini_batches = mk_image_minibatch(img_path, 2, train=false, 
#         aug_pipl=pipl)
# ```
# 
# Model structure is:
# [`ResNet50 V2 topology plot created by netron`]
# (https://github.com/KnetML/NNHelferlein.jl/blob/main/docs/src/assets/netron-resnet50v2.png)
# 
# """
#  function get_mobilenetv3_small(; filters_only=false, trainable=true)
#  
#      local_file = download_pretrained(MOBILENETV3_S_NAME, MOBILENETV3_S_FILE_NAME)
#      h5 = HDF5.h5open(local_file)
#  
#      filter_layers = Chain(
#          Conv(h5, "Conv", use_bias=false, trainable=trainable, stride=2, actf=identity),
#          BatchNorm(h5, "conv2_block1_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
#          Relu(),
#          ResNetBlock([
#                      Conv(h5, "conv2_block1_1_conv", use_bias=false, trainable=trainable, actf=identity),
#                      BatchNorm( h5, "conv2_block1_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
#                      Relu(),
#                      Pad(1),
#                      Conv(h5, "conv2_block1_2_conv", use_bias=false, trainable=trainable, actf=identity),
#                      BatchNorm( h5, "conv2_block1_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
#                      Relu(),
#                      Conv(h5, "conv2_block1_3_conv", trainable=trainable, actf=identity),
#                      ];
#                  shortcut=[
#                      Conv(h5, "conv2_block1_0_conv", trainable=trainable, actf=identity)
#                      ]),
#          ResNetBlock([
#                      BatchNorm(h5, "conv2_block2_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
#                      Relu(),
#                      Conv(h5, "conv2_block2_1_conv", use_bias=false, trainable=trainable, actf=identity),
#                      BatchNorm( h5, "conv2_block2_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
#                      Relu(),
#                      Pad(1),
#                      Conv(h5, "conv2_block2_2_conv", use_bias=false, trainable=trainable, actf=identity),
#      