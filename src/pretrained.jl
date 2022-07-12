# API for tensorflow pre-rained networks
#

const PRETRAINED_DIR = joinpath(NNHelferlein.DATA_DIR, "pretrained")

# download pretrained networks from zenodo:
#
# https://github.com/keras-team/keras/blob/v2.9.0/keras/applications/vgg16.py#L43
# https://storage.googleapis.com/tensorflow/

const VGG16_NAME = "VGG16 (pretrained, TF/Keras)"
const VGG16_KERAS_URL = "https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
const VGG16_FILE_NAME = "vgg16.h5"

const VGG19_NAME = "VGG19 (pretrained, TF/Keras)"
const VGG19_KERAS_URL = "https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5"
const VGG19_FILE_NAME = "vgg19.h5"

const RESNETV2 = "Resnet v2 (pretrained, TF/Keras)"
const RESNETV2_KERAS_URL = "https://storage.googleapis.com/tensorflow/keras-applications/inception_resnet_v2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5"
const RESNETV2_FILE_NAME = "resnetv2.h5"

function download_pretrained(url, name, file_name)

    if isfile(file_name)
        println("Using already downloaded weights for $name")
    else
        println("Downloading weights for $name from Tensorflow storage"); flush(stdout)

        if !isdir(PRETRAINED_DIR)
            mkpath(PRETRAINED_DIR)
        end

        Downloads.download(url, file_name)
    end
end



""" 
    function get_vgg16(; filters_only=false, trainable=true)

Return a VGG16 model with pretrained parameters. 
Parameters have been exported from the Keras application. For details about original model 
and training see
`https://keras.io/api/applications/`.

Model structure is:

![vgg structure](assets/netron-vgg16.png "VGG16")

### Arguments
+ `filters_only=false`: if `true`, only the filterstack is returned   
            (without Flatten() and classifier) to be integrated in to 
            any chain.
+ `trainable=true`: if `true`, the filterstack is set trainable, otherwise
            only the classifier part is trainable and the filter weights are 
            fixed.

"""
function get_vgg16(; filters_only=false, trainable=true)

    local_file = joinpath(PRETRAINED_DIR, VGG16_FILE_NAME)
    download_pretrained(VGG16_KERAS_URL, VGG16_NAME, local_file)
    h5 = HDF5.h5open(local_file)

    filter_layers = Chain(
            Conv(h5, "block1_conv1/block1_conv1_W_1:0", "block1_conv1/block1_conv1_b_1:0", trainable=trainable, padding=1),
            Conv(h5, "block1_conv2/block1_conv2_W_1:0", "block1_conv2/block1_conv2_b_1:0", trainable=trainable, padding=1),
            Pool(),
            Conv(h5, "block2_conv1/block2_conv1_W_1:0", "block2_conv1/block2_conv1_b_1:0", trainable=trainable, padding=1),
            Conv(h5, "block2_conv2/block2_conv2_W_1:0", "block2_conv2/block2_conv2_b_1:0", trainable=trainable, padding=1),
            Pool(),
            Conv(h5, "block3_conv1/block3_conv1_W_1:0", "block3_conv1/block3_conv1_b_1:0", trainable=trainable, padding=1),
            Conv(h5, "block3_conv2/block3_conv2_W_1:0", "block3_conv2/block3_conv2_b_1:0", trainable=trainable, padding=1),
            Conv(h5, "block3_conv3/block3_conv3_W_1:0", "block3_conv3/block3_conv3_b_1:0", trainable=trainable, padding=1),
            Pool(),
            Conv(h5, "block4_conv1/block4_conv1_W_1:0", "block4_conv1/block4_conv1_b_1:0", trainable=trainable, padding=1),
            Conv(h5, "block4_conv2/block4_conv2_W_1:0", "block4_conv2/block4_conv2_b_1:0", trainable=trainable, padding=1),
            Conv(h5, "block4_conv3/block4_conv3_W_1:0", "block4_conv3/block4_conv3_b_1:0", trainable=trainable, padding=1),
            Pool(),
            Conv(h5, "block5_conv1/block5_conv1_W_1:0", "block5_conv1/block5_conv1_b_1:0", trainable=trainable, padding=1),
            Conv(h5, "block5_conv2/block5_conv2_W_1:0", "block5_conv2/block5_conv2_b_1:0", trainable=trainable, padding=1),
            Conv(h5, "block5_conv3/block5_conv3_W_1:0", "block5_conv3/block5_conv3_b_1:0", trainable=trainable, padding=1),
            Pool())

     classif_layers = Chain(
            Dense(h5, "fc1/fc1_W_1:0", "fc1/fc1_b_1:0", trainable=true, actf=relu),
            Dense(h5, "fc2/fc2_W_1:0", "fc2/fc2_b_1:0", trainable=true, actf=relu),
            Dense(h5, "predictions/predictions_W_1:0", "predictions/predictions_b_1:0", trainable=true, actf=identity)) 

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
