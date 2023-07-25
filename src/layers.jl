# Basic layer defs
#
# (c) A. Dominik, 2020



"""
    struct Dense  <: AbstractLayer

Default Dense layer.

### Constructors:
+ `Dense(w, b, actf)`: default constructor, `w` are the weights and `b` the bias.
+ `Dense(i::Int, j::Int; actf=sigm, init=..)`: layer of `j` neurons with
        `i` inputs. Initialiser is xavier_uniform for  `actf=sigm` and
        xaview_normal otherwise.
+ `Dense(h5::HDF5.File, group::String; trainable=false, actf=sigm)`: kernel and bias are loaded by the specified `group`.
+ `Dense(h5::HDF5.File, kernel::String, bias::String;
        trainable=false, actf=sigm)`: layer
        imported from a hdf5-file from TensorFlow with the
        hdf-object h5 and the group name group.
"""
struct Dense  <: AbstractLayer
    w
    b
    actf
    Dense(w, b, actf) = new(w, b, actf)
    Dense(i::Int, j::Int; actf=Knet.sigm, init=((actf==sigm) ? xavier_uniform : xavier_normal)) = 
        new(Knet.param(j,i, init=init), Knet.param0(j), actf)
 end


function Dense(h5::HDF5.File, kernel::String, bias::String; trainable=false, actf=Knet.sigm)

    w = read(h5, kernel)
    b = read(h5, bias)

    w = ifgpu(w)
    b = ifgpu(b)

    if trainable
        w = Param(w)
        b = Param(b)
    end

    (o, i) = size(w)
    println("Generating $actf Dense layer from hdf with $o neurons and $i fan-in.")
    return Dense(w, b, actf)
end
    

function Dense(h5::HDF5.File, group::String; trainable=false, actf=Knet.sigm, tf=true) 
    
    if tf 
        w_path = "model_weights/$group/$group/kernel:0"
        b_path = "model_weights/$group/$group/bias:0"
    else
        w_path = "$group/kernel:0"
        b_path = "$group/bias:0"
    end

    return Dense(h5, w_path, b_path, trainable=trainable, actf=actf)
end


(l::Dense)(x) = l.actf.(l.w * x .+ l.b)

function Base.summary(l::Dense; indent=0)
    n = get_n_params(l)
    o,i = size(l.w)
    s1 = "Dense layer $i → $o with $(l.actf),"
    println(print_summary_line(indent, s1, n))
    return 1
end


"""
    struct Linear  <: AbstractLayer

Almost standard dense layer, but functionality inspired by
the TensorFlow-layer:
+ capable to work with input tensors of
  any number of dimensions
+ default activation function `identity`
+ optionally without biases.

The shape of the input tensor is preserved; only the size of the
first dim is changed from in to out.

### Constructors:
+ `Linear(i::Int, j::Int; bias=true, actf=identity, init=xaview_normal)` 
        where `i` is fan-in and `j` is fan-out.

### Keyword arguments:
+ `bias=true`: if false biases are fixed to 0.0
+ `actf=identity`: activation function.
"""
struct Linear  <: AbstractLayer
    w
    b
    actf
    Linear(w::Param, b::Param, actf) = new(w, b, actf)
    Linear(i::Int, j::Int; bias=true, actf=identity, init=xavier_normal) = 
        new(Knet.param(j,i, init=init), bias ? Knet.param0(j) : init0(j), actf)
 end

 function (l::Linear)(x)
     j,i = size(l.w)   # get fan-in and out
     siz = vcat(j, collect(size(x)[2:end]))
     x = reshape(x, i,:)
     y = l.actf.(l.w * x .+ l.b)
     return reshape(y, siz...)
 end

function Base.summary(l::Linear; indent=0)
    n = get_n_params(l)
    o,i = size(l.w)
    s1 = "Linear layer $i → $o, with $(l.actf),"
    println(print_summary_line(indent, s1, n))
    return 1
end


"""
    struct FeatureSelection  <: AbstractLayer

Simple feature selection layer that maps input to output with
one-by-one connections; i.e. a layer of size 128 has 128 weights
(plus optional biases).

Biases and activation functions are disabled by default.

### Constructors:
+ `FeatureSelection(i; bias=false, actf=identity)`: with the same
            input- and output-size `i`, whre `i` is an integer
            or a Tuple of the input dimensions.

"""
struct FeatureSelection  <: AbstractLayer
    w
    b
    actf
    FeatureSelection(w::Param, b::Param, actf) = new(w, b, actf)
    FeatureSelection(i; bias=false, actf=identity) = 
                new(param(i...), bias ? Knet.param0(i...) : init0(i...), actf)
end

(l::FeatureSelection)(x) = l.actf.(l.w .* x .+ l.b)


function Base.summary(l::FeatureSelection; indent=0)
    n = get_n_params(l)
    i = size(l.w)
    s1 = "Feature selection layer $i → $i, with $(l.actf),"
    println(print_summary_line(indent, s1, n))
    return 1
end

"""
    struct Conv  <: AbstractLayer

Default Conv layer.

### Constructors:
+ `Conv(w1::Int, w2::Int,  i::Int, o::Int; actf=relu; kwargs...)`: layer with
    o kernels of size (w1,w2) for an input of i channels.
+ `Conv(w1::Int, w2::Int, w3::Int, i::Int, o::Int; actf=relu; kwargs...)`: layer 
        with 3-dimensional kernels for 3D convolution 
        (requires 5-dimensional input)
+ `Conv(w1::Int,  i::Int, o::Int; actf=relu; kwargs...)`: layer with
    o kernels of size (1,w1) for an input of i channels.
    This 1-dimensional convolution uses a 2-dimensional kernel with a first 
    dimension of size 1. Input and output contain an empty firfst dimension
    of size 1. If `padding`, `stride` or `dilation` are specified, 2-tuples
    must be specified to correspond with the 2-dimensional kernel
    (e.g. `padding=(0,1)` for a 1-padding along the 1D sequence).

### Constructors to read parameters from Tensorflow/Keras HDF-files:
+ `Conv(h5::HDF5.File, kernel::String, bias::String; trainable=false, actf=Knet.relu, 
   use_bias=true, kwargs...)`:
        Import parameters from HDF file `h5` with `kernel` and `bias` specifying
        the full path to weights and biases, respectively.
+ `Conv(h5::HDF5.File, group::String; trainable=false, actf=relu, tf=true, use_bias=true)`:
        Import a conv-layer from a default TF/Keras HDF5 file. 
        If `tf=false`, `group` defines the full path to the parameters
        `group/kernel:0` and `group/bias:0`. 
        If `tf=true`, `group` defines the  only the group name and 
        parameters are addressed as `model_weights/group/group/kernel:0` and
        `model_weights/group/group/bias:0`.
        

### Keyword arguments:
+ `padding=0`: the number of extra zeros implicitly concatenated
        at the start and end of each dimension.
+ `stride=1`: the number of elements to slide to reach the next filtering window.
+ `dilation=1`: dilation factor for each dimension.
+ `...` See the Knet documentation for Details:
        https://denizyuret.github.io/Knet.jl/latest/reference/#Convolution-and-Pooling.
        All keywords to the Knet function `conv4()` are supported.
"""
struct Conv  <: AbstractLayer
    w
    b
    actf
    kwargs
    Conv(w::Param, b::Param, actf::Function, kwargs) = new(w, b, actf, kwargs)
    Conv(w, b, actf; kwargs...) = new(w, b, actf, kwargs)
    Conv(w1::Int, i::Int, o::Int; actf=Knet.relu, kwargs...) =
            new(Knet.param(1,w1,i,o; init=xavier_normal), Knet.param0(1,1,o,1),
                actf, kwargs)
    Conv(w1::Int, w2::Int,  i::Int, o::Int; actf=Knet.relu, kwargs...) =
            new(Knet.param(w1,w2,i,o; init=xavier_normal), Knet.param0(1,1,o,1),
                actf, kwargs)
    Conv(w1::Int, w2::Int, w3::Int, i::Int, o::Int; actf=Knet.relu, kwargs...) =
            new(Knet.param(w1,w2,w3,i,o; init=xavier_normal), Knet.param0(1,1,1,o,1),
                actf, kwargs)
end

# TODO: bias optional!

(c::Conv)(x) = c.actf.(Knet.conv4(c.w, x; c.kwargs...) .+ c.b)

function Conv(h5::HDF5.File, group::String; trainable=false, actf=Knet.relu, 
              tf=true, use_bias=true, kwargs...)
        if tf 
            kernel = "model_weights/$group/$group/kernel:0"
            bias = "model_weights/$group/$group/bias:0"
        else
            kernel = "$group/kernel:0"
            bias = "$group/bias:0"
        end

        return Conv(h5, kernel, bias; trainable=trainable, actf=actf, use_bias=use_bias, kwargs...)
    end

function Conv(h5::HDF5.File, kernel::String, bias::String; trainable=false, actf=Knet.relu, use_bias=true, kwargs...)

    w = read(h5, kernel)
    #w = permutedims(w, [4,3,2,1])
    w = permutedims(w, [3,4,2,1])  # transpose filters
    w = ifgpu(w)
    if trainable
        w = Param(w)
    end

    siz = size(w)  
    i,o = siz[end-1:end]
    w_siz = siz[1:end-2]

    if use_bias
        b = read(h5, bias)
        b = reshape(b, 1,1,:,1)
        if trainable
            b = param(b)
        else
            b = ifgpu(b)
        end
    else
        b = zeros(Float32, 1,1,o,1) |> ifgpu
    end

    println("Generating Conv layer from hdf with kernel $w_siz, $i channels, $o kernels.")

    return Conv(w, b, actf; mode=1, kwargs...)
end

function Base.summary(l::Conv; indent=0)
    n = get_n_params(l)

    siz = size(l.w)  
    i,o = siz[end-1:end]
    w_siz = siz[1:end-2]
    
    if length(l.kwargs) > 0
        kwa = " $(collect(l.kwargs))"
    else
        kwa = ""
    end
    s1 = "Conv layer $i → $o ($w_siz) $kwa with $(l.actf),"

    println(print_summary_line(indent, s1, n))
    return 1
end


 """
    DepthwiseConv  <: AbstractLayer

Conv layer with seperate filters per input channel. 
*o* output feature maps will be created by performing a convolution 
on only one input channel. `o` must be a multiple of `i`.

### Constructors:
+ `DepthwiseConv(w, b, actf; kwargs)`: default constructor
+ `Conv(w1::Int, w2::Int,  i::Int, o::Int; actf=relu, kwargs...)`: layer with
    `o` kernels of size (w1,w2) for every input channel of an 2-d input of `i` layers.
    `o` must be a multiple of `i`; if `o == i`, each output feature map is 
    generated from one channel. If `o == n*i`, `n` feature maps are 
    generated from each channel.    

### Keyword arguments:
+ `padding=0`: the number of extra zeros implicitly concatenated
        at the start and end of each dimension.
+ `stride=1`: the number of elements to slide to reach the next filtering window.
+ `dilation=1`: dilation factor for each dimension.
"""
struct DepthwiseConv  <: AbstractLayer
    w
    b
    actf
    groups
    kwargs
    
    function DepthwiseConv(w, b, actf, 
                           groups::Int; kwargs...)
        depthwise_warn()
        new(w, b, actf, groups, kwargs)
    end

    function DepthwiseConv(w1::Int, w2::Int, i::Int, o::Int; actf=Knet.relu, 
                            kwargs...) 
        
        depthwise_warn()
        DepthwiseConv(Knet.param(w1,w2,1,o; init=xavier_normal), Knet.param0(1,1,o,1),
                actf,  i; kwargs...)
    end
end

function (c::DepthwiseConv)(x)
    
    if depthwise_warn()
        return x
    else
        return c.actf.(Knet.conv4(c.w, x; group=c.groups, c.kwargs...) .+ c.b)
    end
end


function depthwise_warn()
    if !CUDA.functional()
        println("Grouped convolutions (DepthwiseConv) are not yet supported on CPU!")
        println("As long as there is no functional CUDA backend available, the layer will be ignored!")
        return true
    else
        return false
    end
end

function Base.summary(l::DepthwiseConv; indent=0)
    n = get_n_params(l)

    siz = size(l.w)  
    o = siz[end]
    
    w_siz = siz[1:end-2]
    
    if length(l.kwargs) > 0
        kwa = " $(collect(l.kwargs))"
    else
        kwa = ""
    end
    s1 = "DepthwiseConv layer → $o ($w_siz) in $(l.groups) groups $kwa with $(l.actf),"
    println(print_summary_line(indent, s1, n))
    return 1
end




struct DeConvUnet <: AbstractLayer
    w
    b
    actf
    kwargs
    DeConvUnet(w::Param, b::Param, actf::Function, kwargs) = new(w, b, actf, kwargs)
    DeConvUnet(w, b, actf; kwargs...) = new(w, b, actf, kwargs)
    DeConvUnet(w1::Int, w2::Int,  i::Int, i_enc::Int, o::Int; actf=Knet.relu, kwargs...) =
            new(Knet.param(w1,w2,i+i_enc,o; init=xavier_normal), Knet.param0(1,1,o,1),
                actf, kwargs)
end

function (c::DeConvUnet)(x, enc)
    
    # crop featuremaps to make encoder maps and decoder maps the same dims:
    #
    x = crop_array(x, (size(enc,1), size(enc,2),:,:))
    enc = crop_array(enc, (size(x,1), size(x,2),:,:))
    cat(x, enc, dims=3)

    return c.actf.(Knet.conv4(c.w, x; c.kwargs...) .+ c.b)
end

    

"""
    struct Pool <: AbstractLayer

Pooling layer.

### Constructors:
+ `Pool(;kwargs...)`: max pooling; without `kwargs`, 2-pooling
        is performed.

### Keyword arguments:
+ `window=2`: pooling `window` size (same for all directions)
+ `...`: See the Knet documentation for Details:
        https://denizyuret.github.io/Knet.jl/latest/reference/#Convolution-and-Pooling.
        All keywords to the Knet function `pool` are supported.
"""
struct Pool    <: AbstractLayer
    kwargs
    Pool(kwargs) = new(kwargs)
    Pool(;kwargs...) = new(kwargs)
end

(l::Pool)(x) = Knet.pool(x; l.kwargs...)


function Base.summary(l::Pool; indent=0)
    n = get_n_params(l)
    if length(l.kwargs) > 0
        kwa = " $(collect(l.kwargs))"
    else
        kwa = ""
    end
    s1 = "Pool layer$kwa,"
    println(print_summary_line(indent, s1, n))
    return 1
end



"""
    struct Pad     <: AbstractLayer
    
Pad an n-dimensional array along `dims` with one of the types
supported by `Flux.NNlib`.

### Constructors:
+ `Pad(padding::Int; type=:zeros, dims=nothing)`: Pad with `padding`
            along all dims.

### Keyword arguments:
+ `type`: one of 
    * `:zeros`: zero-padding
    * `:ones`: one-padding
    * `:repeat`: repeat values on the border
    * `:relect`: reflect values across the border
+ `dims`: Tuple of dims to be padded. If `dims==nothing` 
    all except of the last 2 dimensions (i.e. channel and 
    minibatch dimension for convolution layers) are padded.
"""
struct Pad     <: AbstractLayer
    padding
    type
    dims
    Pad(padding, type, dims) = new(padding, type, dims)
    Pad(padding::Int; type=:zeros, dims=nothing) = new(padding, type, dims)
end

function (l::Pad)(x) 
    
    if isnothing(l.dims)
        dims = Tuple(i for i in 1:ndims(x)-2)
    else
        dims=l.dims
    end

    if l.type == :ones
        return NNlib.pad_constant(x, l.padding, 1.0, dims=dims)
    elseif l.type == :repeat
        return NNlib.pad_repeat(x, l.padding, dims=dims)
    elseif l.type == :reflect
        return NNlib.pad_reflect(x, l.padding, dims=dims)
    else # type == :zeros
        return NNlib.pad_zeros(x, l.padding, dims=dims)
    end
end



function Base.summary(l::Pad; indent=0)
    
    s1 = "Padding layer: padding=$(l.padding), $(l.type),"
    println(print_summary_line(indent, s1, 0))
    return 1
end






"""
    struct DeConv  <: AbstractLayer

Default deconvolution layer.

### Constructors:
+ `DeConv(w, b, actf, kwargs...)`: default constructor
+ `DeConv(w1::Int, w2::Int,  i::Int, o::Int; actf=relu, kwargs...)`: layer with
    o kernels of size (w1,w2) for an input of i channels.
+ `DeConv(w1::Int, w2::Int, w3::Int, i::Int, o::Int; actf=relu, kwargs...)`: layer with
    o kernels of size (w1,w2,w3) for an input of i channels.

### Keyword arguments:
+ `padding=0`: the number of extra zeros implicitly concatenated
        at the start and end of each dimension (applied to the output).
+ `stride=1`: the number of elements to slide to reach the next filtering window
        (applied to the output).
+ `...` See the Knet documentation for Details:
        https://denizyuret.github.io/Knet.jl/latest/reference/#Convolution-and-Pooling.
        All keywords to the Knet function `deconv4()` are supported.

"""
struct DeConv  <: AbstractLayer
    w
    b
    actf
    kwargs
    DeConv(w::Param, b::Param, actf, kwargs) = new(w, b, actf, kwargs)
    DeConv(w, b, actf; kwargs...) = new(w, b, actf, kwargs)
    DeConv(w1::Int, w2::Int,  i::Int, o::Int; actf=Knet.relu, kwargs...) =
            new(Knet.param(w1,w2,o,i; init=xavier_normal), Knet.param0(1,1,o,1),
            actf, kwargs)
    DeConv(w1::Int, w2::Int, w3::Int, i::Int, o::Int; actf=Knet.relu, kwargs...) =
            new(Knet.param(w1,w2,w3,o,i; init=xavier_normal), Knet.param0(1,1,1,o,1),
            actf, kwargs)
end

(c::DeConv)(x) = c.actf.(Knet.deconv4(c.w, x; c.kwargs...) .+ c.b)

function Base.summary(l::DeConv; indent=0)
    n = get_n_params(l)
    
    siz = size(l.w)  
    i,o = siz[end-1:end]
    w_siz = siz[1:end-2]
    
    if length(l.kwargs) > 0
        kwa = " $(collect(l.kwargs))"
    else
        kwa = ""
    end
    s1 = "DeConv layer $o → $i ($w_siz) $kwa with $(l.actf),"
    println(print_summary_line(indent, s1, n))
    return 1
end



"""
    struct UnPool <: AbstractLayer

Unpooling layer.

### Constructors:
+ `UnPool(;kwargs...)`: user-defined unpooling
"""
struct UnPool <: AbstractLayer
    kwargs
    UnPool(kwargs::Param) = new(kwargs)
    UnPool(;kwargs...) = new(kwargs)
end
(l::UnPool)(x) = Knet.unpool(x; l.kwargs...)

function Base.summary(l::UnPool; indent=0)
    n = get_n_params(l)
    if length(l.kwargs) > 0
        kwa = " $(collect(l.kwargs))"
    else
        kwa = ""
    end
    s1 = "UnPool layer$kwa,"

    println(print_summary_line(indent, s1, n))
    return 1
end




"""
    struct Flat <: AbstractLayer

Default flatten layer.

### Constructors:
+ `Flat()`: with no options.
"""
struct Flat <: AbstractLayer
end
(l::Flat)(x) = Knet.mat(x)


function Base.summary(l::Flat; indent=0)
    n = get_n_params(l)
    s1 = "Flatten layer,"

    println(print_summary_line(indent, s1, n))
    return 1
end




"""
    struct PyFlat <: AbstractLayer

Flatten layer with optional Python-stype flattening (row-major).
This layer can be used if pre-trained weight matrices from
tensorflow are applied after the flatten layer.

### Constructors:
+ `PyFlat(; python=true)`: if true, row-major flatten is performed.
"""
struct PyFlat <: AbstractLayer
    python
    PyFlat(python::Bool) = new(python)
    PyFlat(; python=true) = new(python)
end
(l::PyFlat)(x) = l.python ? Knet.mat(permutedims(x, (3,2,1,4))) : mat(x)

function Base.summary(l::PyFlat; indent=0)
    n = get_n_params(l)
    if l.python
        s1 = "PyFlat layer with row-major (Python) flattening,"
    else
        s1 = "PyFlat layer with column-major (Julia) flattening,"
    end

    println(print_summary_line(indent, s1, n))
    return 1
end



"""
    struct Embed <: AbstractLayer

Simple type for an embedding layer to embed a virtual onehot-vector
into a smaller number of neurons by linear combination.
The onehot-vector is virtual, because not the vector, but only
the index of the "one" in the vector has to be provided as Integer value
(or a minibatch of integers) with values between 1 and the vocab size.

### Constructors:
+ `Embed(v,d; actf=identity, mask=nothing):` with
    vocab size `v`, embedding depth `d` and default activation function identity.
    `mask` defines the padding token (see below).

### Signatures:
+ `(l::Embed)(x)`: default
  embedding of input tensor `x`.

### Value:
The embedding is constructed by adding a first dimension to the input tensor
with number of rows = embedding depth.
If `x` is a column vector, the value is a matrix. If `x` is as row-vector or
a matrix, the value is a 3-d array, etc.

### Padding and masking:
If a token value is defined as `mask`, occurences are embedded as zero vector.
This can be used for padding sequence with zeros. The masking/padding
token counts to the vocab size. If padding tokens are not masked, their embedding
will be optimised during training (which is not recommended but still possible
for many applications).

Zero may be used as padding token, but it must count to the vocab size 
(i.e. the vocab size must be one larger than the number of tokens)
and the keyword arg `mask=0` must be specified.
"""
struct Embed <: AbstractLayer
    w
    actf
    mask
    Embed(w::Param, actf::Function, mask) = new(w, actf, mask)
    function Embed(i, embed; actf=identity, mask=nothing)
        w = Knet.param(embed,i)
        return new(w, actf, mask)
    end
end

function (l::Embed)(x)

    # fix indices to be 1-based if 0 is used for padding:
    #
    if !isnothing(l.mask) && l.mask == 0
        x = x .+ 1
        mask_token = l.mask+1
    else
        mask_token = l.mask
    end

    y = l.actf.(l.w[:,x])

    if !isnothing(l.mask)
        mask = ifgpu(x .!= mask_token)
        mask = reshape(mask, 1, size(mask)...)
        y = y .* mask
    end
    return y
end


function Base.summary(l::Embed; indent=0)
    n = get_n_params(l)
    o,i = size(l.w)
    s1 = "Embed layer $i → $o, with $(l.actf),"

    println(print_summary_line(indent, s1, n))
    return 1
end



"""
    struct Softmax <: AbstractLayer

Simple softmax layer to compute softmax probabilities.

### Constructors:
+ `Softmax()`
"""
struct Softmax <: AbstractLayer
end
(l::Softmax)(x) = Knet.softmax(x)

function Base.summary(l::Softmax; indent=0)
    n = 0
    s1 = "Softmax layer,"

    println(print_summary_line(indent, s1, n))
    return 1
end

"""
    struct Logistic <: AbstractLayer

Logistic (sigmoid) layer activation with additional
Temperature parameter to control the slope of the curve.
Low temperatures (such as T=0.001) result in a step-like activation 
function,
whereas high temperatures (such as T=10) makes the activation
almoset linear.

### Constructors:
+ `Logistic(; T=1.0)`
"""
struct Logistic <: AbstractLayer
    T
    Logistic(T) = new(T)
    Logistic(;T=1.0) = new(Float32(T))
end
(l::Logistic)(x) = sigm.(x / l.T)

function Base.summary(l::Logistic; indent=0)
    n = 0
    s1 = "Logistic actication layer (T=$(l.T)),"

    println(print_summary_line(indent, s1, n))
    return 1
end




"""
    struct Activation <: AbstractLayer

Simple activation layer with the desired activation function as argument.

### Constructors:
+ `Activation(actf)`
+ `Relu()`
+ `Sigm()`
+ `Swish()`
"""
struct Activation <: AbstractLayer
    actf
end
(l::Activation)(x) = l.actf.(x)

Relu() = Activation(relu)
Sigm() = Activation(sigm)
Swish() = Activation(swish)


function Base.summary(l::Activation; indent=0)
    s1 = "Activation layer, actf=$(l.actf)"

    println(print_summary_line(indent, s1, 0))
    return 1
end



"""
    struct Dropout <: AbstractLayer

Dropout layer.
Implemented with help of Knet's dropout() function that evaluates
AutoGrad.recording() to detect if in training or in prediction.
Dropouts are applied only if prediction.

### Constructors:
+ `Dropout(p)` with the dropout rate *p*.
"""
struct Dropout <: AbstractLayer
    p
end
(l::Dropout)(x) = Knet.dropout(x, l.p)

function Base.summary(l::Dropout; indent=0)
    n = get_n_params(l)
    s1 = "Dropout layer with p = $(l.p),"
    println(print_summary_line(indent, s1, n))
    return 1
end





"""
    struct BatchNorm <: AbstractLayer

Batchnormalisation layer.
Implemented with help of Knet's batchnorm() function that evaluates
AutoGrad.recording() to detect if in training or in prediction.
In training the moments are updated to record the running averages;
in prediction the moments are applied, but not modified.

In addition, optional trainable factor `a` and bias `b` are applied:

```math
y = a \\cdot \\frac{(x - \\mu)}{(\\sigma + \\epsilon)} + b
```

### Constructors:
+ `BatchNorm(; scale=true, channels=0)` will initialise
        the moments with `Knet.bnmoments()` and
        trainable parameters `β` and `γ` only if
        `scale==true` (in this case, the number of channels must
        be defined - for CNNs this is the number of feature maps).

### Constructors to read parameters from Tensorflow/Keras HDF-files:
+ `BatchNorm(h5::HDF5.File, β_path, γ_path, μ_path, var_path; 
                       scale=false, trainable=true, momentum=0.1, ε=1e-5, dims=4)`:
        Import parameters from HDF file `h5` with `β_path`, `γ_path`, 
        `μ_path` and `var_path` specifying
        the full path to β, γ, μ and variance respectively.

+ `BatchNorm(h5::HDF5.File, group::String; scale=false, trainable=true, momentum=0.1, 
                        ε=1e-5, dims=4, tf=true)`:
        Import parameters from HDF file `h5` with parameters in the group
        `group`. Paths to β, γ, μ and variance are constructed 
        if `tf=true` as `model_weights/group/group/beta:0`, etc.
        If `tf=false` group must define the full group path:
        `group/beta:0`.
        `dims` specifies the number of dimensions of the input and may be
        2, 4 or 5. The default (4) applies to standard CNNs 
        (imgsize, imgsize, channels, batchsize).

### Keyword arguments:
+ `scale=true`: if `true`, the trainable scale parameters β and γ
        are used. 
+ `trainable=true`. only used with hdf5-import. If `true` the 
        parameters β and γ are initialised as `Param` and trained in training.

### Details:
2d, 4d and 5d inputs are supported. Mean and variance are computed over
dimensions (2), (1,2,4) and (1,2,3,5) for 2d, 4d and 5d arrays, respectively.

If `scale=true` and `channels != 0`, trainable
parameters `β` and `γ` will be initialised for each channel.

If `scale=true` and `channels == 0` (i.e. `BatchNorm(scale=true)`),
the params `β` and `γ` are not initialised by the constructor.
Instead,
the number of channels is inferred when the first minibatch is normalised
as:
2d: `size(x)[1]`
4d: `size(x)[3]`
5d: `size(x)[4]`
or `0` otherwise.
"""
mutable struct BatchNorm <: AbstractLayer
    scale
    moments
    params
    ε
    BatchNorm(s, m, p, ε) = new(s, m, p, ε)

    function BatchNorm(; scale=true, channels=0, ε=1e-5)
        if scale
            p = init_bn_params(channels)
        else
            p = nothing
        end
        return new(scale, Knet.bnmoments(), p, ε)
    end

    function BatchNorm(h5::HDF5.File, β_path, γ_path, μ_path, var_path; 
                       scale=true, trainable=false, momentum=0.1, ε=1e-5, dims=4)

        μ = read(h5, μ_path)
        var = read(h5, var_path)

        channels = length(μ)
        shape = ones(Int, dims)
        shape[dims-1] = channels

        μ = reshape(μ, shape...) |> ifgpu
        var = reshape(var, shape...) |> ifgpu

        bn_moments = bnmoments(momentum=momentum, mean=μ, var=var)

        if scale
            γ = read(h5, γ_path)
            β = read(h5, β_path)
        
            bn_params = vcat(γ, β) 
            if trainable    # param casts to CuArray{Float32} if in GPU context:
                bn_params = param(bn_params)
            else
                bn_params = ifgpu(bn_params)
            end
        else
            bn_params = nothing
        end

        return new(scale, bn_moments, bn_params, ε)
    end

    function BatchNorm(h5::HDF5.File, group::String; scale=true, trainable=false, momentum=0.1, 
                        ε=1e-5, dims=4, tf=true) 
        
        if tf 
            β_path = "model_weights/$group/$group/beta:0"
            γ_path = "model_weights/$group/$group/gamma:0"
            μ_path = "model_weights/$group/$group/moving_mean:0"
            var_path = "model_weights/$group/$group/moving_variance:0"
        else
            β_path = "$group/beta:0"
            γ_path = "$group/gamma:0"
            μ_path = "$group/moving_mean:0"
            var_path = "$group/moving_variance:0"
        end

        if scale
            println("Generating scaled BatchNorm layer from hdf.")
        else
            println("Generating non-scaled BatchNorm layer from hdf.")
        end
        return BatchNorm(h5, β_path, γ_path, μ_path, var_path;
                  scale=scale, trainable=trainable, momentum=momentum, ε=ε, dims=dims)
    end
end


function (l::BatchNorm)(x)
    if l.scale
        if length(l.params) == 0
            l.params = init_bn_params(x)
        end

        return Knet.batchnorm(x, l.moments, l.params; eps=l.ε)
    else
        return Knet.batchnorm(x, l.moments; eps=l.ε)
    end
end

function init_bn_params(x)

    if x isa Int
        channels = x
    elseif x isa AbstractArray
        dims = size(x)
        if length(dims) in (2, 4, 5)
            channels = dims[end-1]
        else
            channels = 0
        end
    else
        channels = 0
    end
    p = Knet.bnparams(Float32, channels) |> ifgpu
    return p
end

function Base.summary(l::BatchNorm; indent=0)
    n = get_n_params(l)
    if l.scale
        s1 = "Scaled BatchNorm layer,"
    else
        s1 = "Unscaled BatchNorm layer,"
    end
    println(print_summary_line(indent, s1, n))
    return 1
end



"""
    struct LayerNorm  <: AbstractLayer

Simple layer normalisation (inspired by TFs LayerNormalization).
Implementation is from Deniz Yuret's answer to feature request
429 (https://github.com/denizyuret/Knet.jl/issues/492).

The layer performs a normalisation within each sample, *not* batchwise.
Normalisation is modified by two trainable parameters `a` and `b`
(variance and mean)
added to every value of the sample vector.

### Constructors:
+ `LayertNorm(depth; eps=1e-6)`:  `depth` is the number
        of activations for one sample of the layer.

### Signatures:
+ `function (l::LayerNorm)(x; dims=1)`: normalise `x` along the given dimensions.
        The size of the specified dimension must fit with the initialised `depth`.
"""
struct LayerNorm  <: AbstractLayer
    a
    b
    ϵ
end

function LayerNorm(depth; eps=1e-6)
        a = param(depth; init=ones)
        b = param(depth; init=zeros)
        LayerNorm(a, b, Float32(eps))
end

function (l::LayerNorm)(x; dims=1)
    μ = mean(x, dims=dims)
    σ = std(x, mean=μ, dims=dims)
    return l.a .* (x .- μ) ./ (σ .+ l.ϵ) .+ l.b
end

function Base.summary(l::LayerNorm; indent=0)
    n = get_n_params(l)
    s1 = "Trainable LayerNorm layer,"

    println(print_summary_line(indent, s1, n))
    return 1
end

"""
    struct GaussianNoise

Gaussian noise layer. Multiplies Gaussian-distributed random values with 
*mean = 1.0* and *sigma = σ* to each training value.

### Constructors:
+ `aussianNoise(σ; train_only=true)`

### Arguments:
+ `σ`: Standard deviation for the distribution of noise
+ `train_only=true`: if `true`, noise will only be applied in training.
"""
struct GaussianNoise <: AbstractLayer
    σ
    train_only
    GaussianNoise(σ::Number, train_only::Bool) = new(σ, train_only)
    GaussianNoise(σ; train_only=true) = new(σ, train_only)
end

function (l::GaussianNoise)(x)

    if AutoGrad.recording() || !l.train_only
        return do_noise(x, l.σ)
    else
        return x
    end
end

function Base.summary(l::GaussianNoise; indent=0)
    s1 = "GaussianNoise layer with σ = $(l.σ)"
    println(print_summary_line(indent, s1, 0))
    return 1
end




"""
    struct GlobalAveragePooling  <: AbstractLayer

Layer to return a matrix with the mean values of all but the last two
dimensions for each sample of the minibatch.
If the input is a stack of feature maps from a convolutional layer,
the result can be seen as the mean value of each feature map.
Number of *output*-rows equals number of *input*-featuremaps; 
number of *output*-columns equals size of minibatch. 

### Constructors:
    GlobalAveragePooling()

"""
struct GlobalAveragePooling  <: AbstractLayer
end

(l::GlobalAveragePooling)(x) = mean(x, dims=(Tuple(collect(1:ndims(x))[1:end-2]))) |> x-> reshape(x, size(x)[end-1],:)
#(l::GlobalAveragePooling)(x) = mean(x, dims=(1,2)) |> x-> reshape(x, size(x)[end-1],:)

function Base.summary(l::GlobalAveragePooling; indent=0)
    s1 = "Global average pooling layer"
    println(print_summary_line(indent, s1, 0))
    return 1
end


# Recurrent layers:
#
#
#

"""
    struct Recurrent <: AbstractLayer

One layer RNN that works with minibatches of (time) series data.
Minibatch can be a 2- or 3-dimensional Array.
If 2-d, inputs for one step are in one column and the Array has as
many colums as steps.
If 3-d, the last dimension iterates the samples of the minibatch.

Result is an array matrix with the output of the units of all
steps for all smaples of the minibatch (with model depth as first and samples of the minimatch as last dimension).

### Constructors:

    Recurrent(n_inputs::Int, n_units::Int; u_type=:lstm, 
              bidirectional=false, allow_mask=false, o...)

+ `n_inputs`: number of inputs
+ `n_units`:  number of units 
+ `u_type` :  unit type can be one of the Knet unit types
        (`:relu, :tanh, :lstm, :gru`) or a type which must be a 
        subtype of `RecurrentUnit` and fullfill the respective interface 
        (see the docs for `RecurentUnit`).
+ `bidirectional=false`: if true, 2 layers of `n_units` units will be defined
        and run in forward and backward direction respectively. The hidden
        state is `[2*n_units*mb]` or `[2*n_units,steps,mb]` id `return_all==true`.
+ `allow_mask=false`: if masking is allowed, a slower algorithm is used to be 
        able to ignore any masked step. Arbitrary sequence positions may be 
        masked for any sequence.
Any keyword argument of `Knet.RNN` or 
a self-defined `RecurrentUnit` type may be provided.

### Signatures:

    function (rnn::Recurrent)(x; c=nothing, h=nothing, return_all=false, 
              mask=nothing)

The layer is called either with a 2-dimensional array of the shape
[fan-in, steps] 
or a 3-dimensional array of [fan-in, steps, batchsize].

#### Arguments:

+ `c=0`, `h=0`: inits the hidden and cell state.
    If `nothing`,  states `h` or `c` keep their values. 
    If `c=0` or `h=0`, the states are resetted to `0`;
    otherwise an array of states of the correct dimensions can be supplied 
    to be used as initial states.
+ `return_all=false`: if `true` an array with all hidden states of all steps 
    is returned (size is [units, time-steps, minibatch]).
    Otherwise only the hidden states of the last step are returned
    ([units, minibatch]).
+ `mask`: optional mask for the input sequence minibatch of shape 
    [steps, minibatch]. Values in the mask must be 1.0 for masked positions
    or 0.0 otherwise and of type `Float32` or `CuArray{Float32}` for GPU context. 
    Appropriate masks can be generated with the NNHelferlein function 
    `mk_padding_mask()`.

Bidirectional layers can be constructed by specifying `bidirectional=true`, if
the unit-type supports it (Knet.RNN does). 
Please be aware that the actual number of units is 2 x n_units for 
bidirectional layers and the output dimension is [2 x units, steps, mb] or
[2 x units, mb].
"""
struct Recurrent <: AbstractLayer
    n_inputs
    n_units
    unit_type
    rnn
    back_rnn
    has_c
    allow_mask

    Recurrent(n_inputs::Int, n_units::Int, unit_type::Symbol, 
              rnn::Knet.RNN, back_rnn::Union{Knet.RNN, Nothing},
              has_c::Bool, allow_mask::Bool) = new(n_inputs, n_units, unit_type, rnn, back_rnn, has_c, allow_mask) 

    function Recurrent(n_inputs::Int, n_units::Int; u_type=:lstm, 
                       allow_mask=false, bidirectional=false, o...)
        back = nothing
        if u_type isa Symbol 
            if !allow_mask
                rnn = Knet.RNN(n_inputs, n_units; rnnType=u_type, h=0, c=0, 
                               bidirectional=bidirectional, o...)
            else
                rnn = Knet.RNN(n_inputs, n_units; rnnType=u_type, h=0, c=0, 
                               bidirectional=false, o...)
                if bidirectional
                    back = Knet.RNN(n_inputs, n_units; rnnType=u_type, h=0, c=0, 
                                    bidirectional=false, o...)
                end
            end

        elseif u_type isa Type && u_type <: RecurrentUnit
            rnn = u_type(n_inputs, n_units; o...)
            if bidirectional
                back = u_type(n_inputs, n_units; o...)
            end
        end
        return new(n_inputs, n_units, u_type, rnn, back,  ## ToDo: back not rnn!
                   hasproperty(rnn, :c), allow_mask)
    end
end


function (rnn::Recurrent)(x; c=0, h=0, 
                          return_all=false, mask=nothing)
    
    if ndims(x) == 3    # x is [fan-in, steps, mb]
        fanin, steps, mb = size(x)
    else
        fanin, steps = size(x)
        mb = 1
        x = reshape(x, fanin, steps, mb)
    end
    @assert fanin == rnn.n_inputs "input does not match the fan-in of rnn layer"

    x = permutedims(x, (1,3,2))   # make [fanin, mb, steps] for Knet
    
    # init h and c fields:
    #
    if !isnothing(h)     # ToDo: unbox old vals! h = value(h)
        rnn.rnn.h = h
        if !isnothing(rnn.back_rnn)
            rnn.back_rnn.h = h
        end
    end
    if rnn.has_c && !isnothing(c)
        rnn.rnn.c = c
        if !isnothing(rnn.back_rnn)
            rnn.back_rnn.c = c
        end
    end

    # life is easy without masking and if Knet.RNN
    # otherwise step-by-step loop is needed:
    #
    if rnn.rnn isa Knet.RNN && !rnn.allow_mask
        #println("Knet")
        hidden = rnn.rnn(x)
    else
        #println("manual")
        if isnothing(rnn.back_rnn)   # not bidirectional:
            hidden = rnn_loop(rnn.rnn, x, rnn.n_units, mask, false, return_all)     
        else        
            h_f = rnn_loop(rnn.rnn, x, rnn.n_units, mask, true , return_all)
            h_r = rnn_loop(rnn.back_rnn, x, rnn.n_units, mask, true, return_all)
            
            if return_all
                hidden = cat(h_f, h_r[:,:,end:-1:1], dims=1)
            else
                hidden = cat(rnn.rnn.h, rnn.back_rnn.h, dims=1)
                hidden = reshape(hidden, 2*rnn.n_units, mb, 1)
            end
        end

    end


    if return_all
        return permutedims(hidden, (1,3,2)) # h of all steps: [units, time-steps, mb]
    else
        return hidden[:,:,end]     # last h: [units, mb]
    end
end



# inner loop for rnn - not exposed!
# x is [fanin, mb, steps]
#
function rnn_loop(rnn, x, n_units, mask=nothing, backward=false,
                  return_all=true)

    fanin, mb, steps = size(x)

    # init h and c fields and mask
    # make sure the size is correct:
    #
    if rnn.h == 0 || isnothing(rnn.h)
        rnn.h = init0(n_units, mb)
    end
    if hasproperty(rnn, :c) && (rnn.c == 0 || isnothing(rnn.c))
            rnn.c = init0(n_units, mb)
    end

    # init h collection:
    #
    if return_all
        hs = emptyKnetArray(n_units, mb, 0)
    end

    if backward
        step_range = steps:-1:1
    else
        step_range = 1:steps
    end
    for i in step_range
        last_h = value(rnn.h)
        last_c = value(rnn.c)

        h_step = rnn(x[:,:,i])               # run one step only

        # h and c with masking:
        #
        if !isnothing(mask)
            # m_step = recycle_array(mask[[i],:], rnn.n_units, dims=1)
            m_step = mask[[i],:]
       
            # restore old c if masked position:
            #
            h_step = last_h .* m_step + h_step .* (1 .- m_step)
            if hasproperty(rnn, :c)
                 rnn.c = last_c .* m_step + rnn.c .* (1 .-m_step)
            end
        end

        rnn.h = h_step
        if return_all
            hs = cat(hs, h_step, dims=3)  
        end
    end
    if !return_all
        hs = reshape(rnn.h, n_units, mb, 1)
    end
        return hs
end

function Base.summary(l::Recurrent; indent=0)
    n = get_n_params(l)
    if isnothing(l.back_rnn)
        s1 = "Recurrent layer, "
    else
        s1 = "Bidirectional Recurrent layer, "
    end
    s1 = s1 * "$(l.n_inputs) → $(l.n_units) of type $(l.unit_type),"
    println(print_summary_line(indent, s1, n))
    return 1
end

"""
    function get_hidden_states(l::<RNN_Type>; flatten=true)

Return the hidden states of one or more layers of an RNN.
`<RNN_Type>` is one of `NNHelferlein.Recurrent`, `Knet.RNN`.

### Arguments:
+ `flatten=true`: if the states tensor is 3d with a 3rd dim > 1, the 
        array is transformed to [units, mb, 1] to represent all current states
        after the last step.
"""
function get_hidden_states(l::Union{Recurrent, Knet.RNN}; flatten=true)
    
    if l isa Recurrent 
        h = l.rnn.h
    elseif l isa Knet.RNN 
        h = l.h
    else
        h = nothing
    end

    if !isnothing(h)
        if flatten && ndims(h) == 3 && size(h)[3] > 1
            units, mb, bi = size(h)
            h = reshape(permutedims(h, (1,3,2)), units*bi, mb, :)
        end
    end
    return h
end

"""
    function get_cell_states(l::<RNN_Type>; unbox=true, flatten=true)

Return the cell states of one or more layers of an RNN only if
it is a LSTM (Long short-term memory).

### Arguments:
+ `unbox=true`: By default, c is unboxed when called in `@diff` context (while AutoGrad 
        is recording) to avoid unwanted dependencies of the computation graph
        s2s.attn(reset=true)
        (backprop should run via the hidden states, not the cell states).
+ `flatten=true`: if the states tensor is 3d with a 3rd dim > 1, the 
        array is transformed to [units, mb, 1] to represent all current states
        after the last step.
"""
function get_cell_states(l::Union{Recurrent, Knet.RNN}; unbox=true, flatten=true)
    if l isa Recurrent 
        c = l.rnn.c
    elseif l isa Knet.RNN 
        c = l.c
    else
        c = nothing
    end

    if !isnothing(c)
        if unbox
            c = value(c)
        end
        if flatten && ndims(c) == 3 && size(c)[3] > 1
            units, mb, bi = size(c)
            c = reshape(permutedims(c, (1,3,2)), units*bi, mb, :)
        end
    end
    return c
end


"""
    function set_hidden_states!(l::<RNN_Type>, h)

Set the hidden states of one or more layers of an RNN
to `h`.
"""
function set_hidden_states!(l::Union{Recurrent, Knet.RNN}, h)
    if l isa Recurrent
        l.rnn.h = h
    elseif l isa Knet.RNN
        l.h = h
    end
end

"""
    function set_cell_states!(l::<RNN_Type>, c)

Set the cell states of one or more layers of an RNN
to `c`.
"""
function set_cell_states!(l::Union{Recurrent, Knet.RNN}, c)
    if l isa Recurrent
        l.rnn.c = c
    elseif l isa Knet.RNN
        l.c = c
    end
end



"""
    function reset_hidden_states!(l::<RNN_Type>)

Reset the hidden states of one or more layers of an RNN
to 0.
"""
function reset_hidden_states!(l::Union{Recurrent, Knet.RNN})
    if l isa Recurrent
        l.rnn.h = 0
    elseif l isa Knet.RNN
        l.h = 0
    end
end

"""
    function reset_cell_states!(l::<RNN_Type>)

Reset the cell states of one or more layers of an RNN
to 0.
"""
function reset_cell_states!(l::Union{Recurrent, Knet.RNN})
    if l isa Recurrent
        l.rnn.c = 0
    elseif l isa Knet.RNN
        l.c = 0
    end
end








# return number of params:
#
function get_n_params(mdl)

    n = 0
    for p in params(mdl)
        n += length(p)
    end
    return n
end
