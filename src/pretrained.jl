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
const RESNET50V2_FILE_NAME = "resnet50v2keras_model.h5"





function download_pretrained(name, file_name)

    url = "$ZENODO_URL_PRETRAINED/$file_name"
    local_file = "$PRETRAINED_DIR/$file_name"

    if isfile(local_file)
        println("Using already downloaded weights for $name")
    else
        println("Downloading weights for $name from Zonodo")
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
`https://keras.io/api/applications/`.

### Arguments
+ `filters_only=false`: if `true`, only the filterstack is returned   
            (without Flatten() and classifier) to be integrated in to 
            any chain.
+ `trainable=true`: if `true`, the filterstack is set trainable, otherwise
            only the classifier part is trainable and the filter weights are 
            fixed.

Model structure is:

![netron](assets/netron-vgg16-w200.png)
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
`https://keras.io/api/applications/`.

### Arguments
+ `filters_only=false`: if `true`, only the filterstack is returned   
            (without Flatten() and classifier) to be integrated in to 
            any chain.
+ `trainable=true`: if `true`, the filterstack is set trainable, otherwise
            only the classifier part is trainable and the filter weights are 
            fixed.

Model structure is:

"""
function get_resnet50v2(; filters_only=false, trainable=true)

    local_file = download_pretrained(RESNET50V2_NAME, RESNET50V2_FILE_NAME)
    h5 = HDF5.h5open(local_file)

    filter_layers = Chain(
        Conv(h5, "conv1_conv", 
            trainable=trainable, padding=3, stride=2, actf=identity),
        Pool(;padding=1, window=3, stride=1),
        BatchNorm(h5, "conv2_block1_preact_bn",
            trainable=trainable, momentum=0.99, ε=1.001e-5),
        Relu(),
        ResNetBlock([
                    Conv(h5, "conv2_block1_1_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    BatchNorm( h5, "conv2_block1_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv2_block1_2_conv", trainable=trainable, padding=1, stride=1, actf=identity),
                    BatchNorm( h5, "conv2_block1_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv2_block1_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ];
                    shortcut=[
                    Conv(h5, "conv2_block1_0_conv", trainable=trainable, padding=0, stride=1, actf=identity)
                    ]),
        ResNetBlock([
                    BatchNorm(h5, "conv2_block2_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv2_block2_1_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    BatchNorm( h5, "conv2_block2_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv2_block2_2_conv", trainable=trainable, padding=1, stride=1, actf=identity),
                    BatchNorm( h5, "conv2_block2_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv2_block2_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ]),
        ResNetBlock([
                    BatchNorm(h5, "conv2_block3_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv2_block3_1_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    BatchNorm( h5, "conv2_block3_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv2_block3_2_conv", trainable=trainable, padding=1, stride=1, actf=identity),
                    BatchNorm( h5, "conv2_block3_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv2_block3_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ],
                    shortcut = [
                    Pool(1, stride=[2,2])
                    ]),
        BatchNorm( h5, "conv3_block1_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
        Relu(),
        ResNetBlock([
                    Conv(h5, "conv3_block1_1_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    BatchNorm(h5, "conv3_block1_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block1_2_conv", trainable=trainable, padding=1, stride=1, actf=identity),
                    BatchNorm( h5, "conv3_block1_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block1_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ],
                    shortcut = [
                    Conv(h5, "conv3_block1_0_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ]),
        ResNetBlock([
                    BatchNorm(h5, "conv3_block2_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block2_1_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    BatchNorm(h5, "conv3_block2_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block2_2_conv", trainable=trainable, padding=1, stride=1, actf=identity),
                    BatchNorm( h5, "conv3_block2_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block2_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ],
                    shortcut = []
                    ),
        ResNetBlock([
                    BatchNorm(h5, "conv3_block3_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block3_1_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    BatchNorm(h5, "conv3_block3_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block3_2_conv", trainable=trainable, padding=1, stride=1, actf=identity),
                    BatchNorm( h5, "conv3_block3_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block3_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ],
                    shortcut = []
                    ),
        ResNetBlock([
                    BatchNorm(h5, "conv3_block4_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block4_1_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    BatchNorm( h5, "conv3_block4_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block4_2_conv", trainable=trainable, padding=1, stride=1, actf=identity),
                    BatchNorm( h5, "conv3_block4_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv3_block4_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ],
                    shortcut = [
                    Pool(1, stride=[2,2])
                    ]),
        BatchNorm( h5, "conv4_block1_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
        Relu(),
        ResNetBlock([
                    Conv(h5, "conv4_block1_1_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    BatchNorm( h5, "conv4_block1_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block1_2_conv", trainable=trainable, padding=1, stride=1, actf=identity),
                    BatchNorm( h5, "conv4_block1_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block1_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ];
                    shortcut=[
                    Conv(h5, "conv4_block1_0_conv", trainable=trainable, padding=0, stride=1, actf=identity)
                    ]),
        ResNetBlock([
                    BatchNorm(h5, "conv4_block2_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block2_1_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    BatchNorm(h5, "conv4_block2_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block2_2_conv", trainable=trainable, padding=1, stride=1, actf=identity),
                    BatchNorm( h5, "conv4_block2_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block2_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ],
                    shortcut = []
                    ),
        ResNetBlock([
                    BatchNorm(h5, "conv4_block3_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block3_1_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    BatchNorm(h5, "conv4_block3_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block3_2_conv", trainable=trainable, padding=1, stride=1, actf=identity),
                    BatchNorm( h5, "conv4_block3_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block3_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ],
                    shortcut = []
                    ),
        ResNetBlock([
                    BatchNorm(h5, "conv4_block4_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block4_1_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    BatchNorm(h5, "conv4_block4_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block4_2_conv", trainable=trainable, padding=1, stride=1, actf=identity),
                    BatchNorm( h5, "conv4_block4_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block4_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ],
                    shortcut = []
                    ),
        ResNetBlock([
                    BatchNorm(h5, "conv4_block5_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block5_1_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    BatchNorm(h5, "conv4_block4_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block5_2_conv", trainable=trainable, padding=1, stride=1, actf=identity),
                    BatchNorm( h5, "conv4_block5_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block5_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ],
                    shortcut = []
                    ),
        ResNetBlock([
                    BatchNorm(h5, "conv4_block6_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block6_1_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    BatchNorm( h5, "conv4_block6_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block6_2_conv", trainable=trainable, padding=1, stride=1, actf=identity),
                    BatchNorm( h5, "conv4_block6_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv4_block6_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ],
                    shortcut = [
                    Pool(1, stride=[2,2])
                    ]),
        BatchNorm( h5, "conv5_block1_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
        Relu(),
        ResNetBlock([
                    Conv(h5, "conv5_block1_1_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    BatchNorm( h5, "conv5_block1_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv5_block1_2_conv", trainable=trainable, padding=1, stride=1, actf=identity),
                    BatchNorm( h5, "conv5_block1_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv5_block1_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ];
                    shortcut=[
                    Conv(h5, "conv5_block1_0_conv", trainable=trainable, padding=0, stride=1, actf=identity)
                    ]),
        ResNetBlock([
                    BatchNorm(h5, "conv5_block2_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv5_block2_1_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    BatchNorm(h5, "conv5_block2_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv5_block2_2_conv", trainable=trainable, padding=1, stride=1, actf=identity),
                    BatchNorm( h5, "conv5_block2_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv5_block2_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ],
                    shortcut = []
                    ),
        ResNetBlock([
                    BatchNorm(h5, "conv5_block3_preact_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv5_block3_1_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    BatchNorm(h5, "conv5_block3_1_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv5_block3_2_conv", trainable=trainable, padding=1, stride=1, actf=identity),
                    BatchNorm( h5, "conv5_block3_2_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
                    Relu(),
                    Conv(h5, "conv5_block3_3_conv", trainable=trainable, padding=0, stride=1, actf=identity),
                    ],
                    shortcut = []
                    ),
        BatchNorm(h5, "post_bn", trainable=trainable, momentum=0.99, ε=1.001e-5),
        Relu()
    )

    classif_layers = Chain(
            Dense(h5, "predictions", trainable=true, actf=identity))

    if filters_only
         vgg = Chain(filter_layers)
    else
        vgg = Classifier( 
            filter_layers,
            GlobalAveragePooling(),
            classif_layers)
    end

    println("")
    println("Imported pretrained network $VGG16_NAME")
    print_network(vgg)
    return vgg
end


struct ResNetBlock <: AbstractChain
    layers      # [1]: conv layers Chain
                # [2]: Chain in the shortcut route
                # [3]: Chain after rejoining 

    ResNetBlock(layers; shortcut=[identity], post=[identity]) = 
            new([Chain(layers...), Chain(shortcut...), Chain(post...)])

    function (b::ResNetBlock)(x)
        return y = b.layers[3](b.layers[1](x) .+ b.layers[2](x))
    end
end

# TODO: summary fpr Resnetblock




    