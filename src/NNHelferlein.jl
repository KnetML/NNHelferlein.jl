module NNHelferlein
# import Pkg; Pkg.add("Images")
import Base: iterate, length, summary
using LinearAlgebra
using Unicode
import ZipFile
import HDF5
import JLD2
using Statistics: mean, std
using ProgressMeter, Dates
using IterTools: ncycle, takenth
import CSV
using DataFrames
import Random
using Printf
import CUDA
# import NNlib
using Knet #: KnetArray, Param, @diff
import Images, Colors
import Augmentor
import MLDataUtils
using MLDatasets: MNIST, FashionMNIST
using TensorBoardLogger, Logging
import MLBase: confusmat
import Downloads
import Adapt
import NNlib
# TODO: tidy-up!

include("types.jl")
include("util.jl")
include("iterators.jl")
include("nets.jl")
include("layers.jl")
include("attn.jl")
include("funs.jl")
include("transformers.jl")
include("images.jl")
include("dataframes.jl")
include("texts.jl")
include("train.jl")
include("acc.jl")
include("imagenet.jl")
include("datasets.jl")
include("io.jl")
include("bioinformatics.jl")

DATA_DIR = normpath(joinpath(dirname(pathof(@__MODULE__)), "..", "data"))
include("pretrained.jl")

export AbstractNN, Classifier, Regressor, AbstractChain, Chain,           # chains
       VAE, get_beta, set_beta!,
       DataLoader, SequenceData, PartialIterator,
       RecurrentUnit,
       add_layer!, 
       split_minibatches,
       ImageLoader, 
       get_class_labels,
       iterate, length,
       AbstractLayer, Dense, Conv, Pool, Flat, PyFlat, flatten        # layers
       Pad,
       FeatureSelection,
       DeConv, UnPool,  DepthwiseConv,
       Embed, Recurrent,
       Softmax, Logistic,
       Dropout, BatchNorm, LayerNorm,
       Linear, GaussianNoise,
       GlobalAveragePooling, global_average_pooling
       Activation, Logistic, Sigm, Relu, Swish,
       ResNetBlock,
       get_hidden_states, get_cell_states,
       set_hidden_states!, set_cell_states!,
       reset_hidden_states!, reset_cell_states!,
       AttentionMechanism, AttnBahdanau,
       AttnLuong, AttnDot, AttnLocation,
       AttnInFeed,
       leaky_sigm, leaky_relu, leaky_tanh,
       PositionalEncoding, positional_encoding_sincos,  # transformers
       Transformer, TokenTransformer,
       TFEncoder, TFDecoder, TFEncoderLayer, TFDecoderLayer,
       mk_padding_mask, mk_peek_ahead_mask,
       dot_prod_attn, MultiHeadAttn,
       separate_heads, merge_heads,
       dataframe_read, dataframe_split,         # import data
       dataframe_minibatches, dataframe_minibatch, mk_class_ids,
       MBNoiser, MBMasquerade, GPUIterator,  
       mk_image_minibatch,
       tb_train!,
       predict_top5, predict_imagenet,
       predict, hamming_dist, hamming_acc,
       peak_finder_acc,
       get_imagenet_classes,                    # images
       image2array, array2image, array2RGB,
       clean_sentence, WordTokenizer,                    # texts
       get_tatoeba_corpus,
       sequence_minibatch, pad_sequence, truncate_sequence, 
       TOKEN_START, TOKEN_END, TOKEN_PAD, TOKEN_UNKOWN,
       crop_array, init0,                               # utils
       convert2CuArray, emptyCuArray, ifgpu,
       convert2KnetArray, emptyKnetArray,
       blowup_array, recycle_array,
       de_embed,
       print_network,
       copy_network, save_network, load_network,
       DATA_DIR, 
       confusion_matrix,
       squared_error_acc, abs_error_acc, minibatch_eval,   # eval
       focal_nll, focal_bce,
       dataset_mit_nsr, dataset_iris, 
       dataset_mnist, dataset_fashion_mnist,
       dataset_pfam,
       get_vgg16, get_resnet50v2, 
       preproc_imagenet_vgg, preproc_imagenet_resnet, preproc_imagenet_resnetv2,
       leaky_sigm, leaky_relu, leaky_tanh, swish,
       aminoacid_tokenizer, embed_blosum62, embed_vhse8,      # bioinformatics
       EmbedAminoAcids

end # module
