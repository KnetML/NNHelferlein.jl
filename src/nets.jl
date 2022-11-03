# Chains for NNs:
#
# (c) A. Dominik, 2021


"""
    abstract type AbstractNN

Mother type for AbstractNN hierarchy with implementation for a chain of layers.

### Signatures:
+ `(m::AbstractNN)(x)`: run the AbstractArray `x` througth all layers and return
                        the output
+ `(m::AbstractNN)(x,y)`: Calculate the loss for one minibatch `x` and teaching input `y`
+ `(m::AbstractNN)(d::Knet.Data)`: Calculate the loss for all minibatches in `d`
+ `(m::AbstractNN)(d::Tuple)`: Calculate the loss for all minibatches in `d`
+ `(m::AbstractNN)(d::NNHelferlein.DataLoader)`: Calculate the loss for all minibatches in `d` 
                        if teaching input is included (i.e. elements of d are tuples).
                        Otherwise return the out of all minibatches as one array with 
                        samples as columns.
```
"""
abstract type AbstractNN
end
(m::AbstractNN)(x) = (for l in m.layers; x = l(x); end; x)
(m::AbstractNN)(x,y) = m(x,y)
function (m::AbstractNN)(d::Union{Tuple, Knet.Data, NNHelferlein.DataLoader}) 
    if first(d) isa AbstractArray # only x without teaching input
        return hcat([m(x) for x in d]...)
    else
        return mean( m(x,y) for (x,y) in d)
    end
end



"""
    abstract type AbstractChain

Mother type for AbstractChain hierarchy with implementation for a chain of layers.
By default every `AbstractChain` has a property `layers` with a iterable list of 
`AbstractLayer`s or `AbstractChain`s that are executed recursively.

Non-standard Chains in which Layers are not execueted sequnetially (such as ResnetBlocks)
must provide a custom implementation with the signature `chain(x)`.

### Signatures:
+ `(m::AbstractChain)(x)`: run the AbstractArray `x` througth all layers and return
                        the output
```
"""
abstract type AbstractChain
end
(m::AbstractChain)(x) = (for l in m.layers; x = l(x); end; x)




"""
    struct Classifier <: AbstractNN

Classifier with default nll loss.
An alternative loss function can be supplied as keyword argument.
The function must provide a signature to be called as 
`loss(model(x), y)`.

### Constructors:
    Classifier(layers...; loss=Knet.nll)

### Signatures:
    (m::Classifier)(x,y) = m.loss(m(x), y)
"""
struct Classifier <: AbstractNN
    layers
    loss
    Classifier(layers::Vector, loss::Function) = new(layers, loss)
    Classifier(layers...; loss=Knet.nll) = new(Any[layers...], loss)
end
(m::Classifier)(x,y) = m.loss(m(x), y)





"""
    struct Regressor

Regression network with square loss as loss function.

### Constructors:
    Regressor(layers...; loss=mean_squared_error.nll)

### Signatures:
    (m::Regression)(x,y) = mean(abs2, Array(m(x)) - y)
"""
struct Regressor <: AbstractNN
    layers
    loss
    Regressor(layers::Vector, loss::Function) = new(layers, loss)
    Regressor(layers...; loss=mean_squared_error) = new(Any[layers...], loss)
end
#(m::Regressor)(x,y) = mean(abs2, ifgpu(y) .- m(x))
(m::Regressor)(x,y) = m.loss(m(x), ifgpu(y))




"""
    struct Chain <: AbstractChain

Simple wrapper to chain layers and execute them one after another.
"""
struct Chain <: AbstractChain
    layers
    Chain(layers::Vector) = new(layers)
    Chain(layers...) = new(Any[layers...])
end

# sequencial interface:
#
import Base: push!, length
push!(n::Union{NNHelferlein.AbstractNN, NNHelferlein.AbstractChain}, l) = push!(n.layers, l)
length(n::Union{NNHelferlein.AbstractNN, NNHelferlein.AbstractChain}) = length(n.layers)

"""
    add_layer!(n::Union{NNHelferlein.AbstractNN, NNHelferlein.AbstractChain}, l)

Add a layer `l` or a chain to a model `n`. The layer is always added 
at the end of the chains. 
The modified model is returned.
"""
function add_layer!(n::Union{NNHelferlein.AbstractNN, NNHelferlein.AbstractChain}, l)
    push!(n.layers, l)
    return n
end


"""
    function +(n::Union{NNHelferlein.AbstractNN, NNHelferlein.AbstractChain}, l::Union{AbstractLayer, AbstractChain})
    function +(l1::AbstractLayer, l2::Union{AbstractLayer, AbstractChain})

The `plus`-operator is overloaded to be able to add layers and chains 
to a network.

The second form returns a new chain if 2 Layers are added.

### Example:

```julia
julia> mdl = Classifier() + Dense(2,5)
julia> print_network(mdl)

NNHelferlein neural network summary:
Classifier with 1 layers,                                           15 params
Details:
 
    Dense layer 2 → 5 with sigm,                                    15 params
 
Total number of layers: 1
Total number of parameters: 15


julia> mdl = mdl + Dense(5,5) + Dense(5,1, actf=identity)
julia> print_network(mdl)

NNHelferlein neural network summary:
Classifier with 3 layers,                                           51 params
Details:
 
    Dense layer 2 → 5 with sigm,                                    15 params
    Dense layer 5 → 5 with sigm,                                    30 params
    Dense layer 5 → 1 with identity,                                 6 params
 
Total number of layers: 3
Total number of parameters: 51
```
"""
function Base.:+(n::Union{NNHelferlein.AbstractNN, NNHelferlein.AbstractChain}, l::Union{NNHelferlein.AbstractLayer, NNHelferlein.AbstractChain})
    add_layer!(n, l)
    return n
end

function Base.:+(l1::NNHelferlein.AbstractLayer, l2::Union{NNHelferlein.AbstractLayer, NNHelferlein.AbstractChain})
    return NNHelferlein.Chain(l1, l2)
end




function Base.summary(mdl::Union{NNHelferlein.AbstractNN, NNHelferlein.AbstractChain}; indent=0)
    n = get_n_params(mdl)
    if hasproperty(mdl, :layers)
        ls = length(mdl.layers)
        s1 = "$(typeof(mdl)) with $ls layers,"
    else
        s1 = "$(typeof(mdl)),"
    end
    return print_summary_line(indent, s1, n)
end


"""
    function print_network(mdl::AbstractNN)

Print a network summary of any model of Type `AbstractNN`.
If the model has a field `layers`, the summary of all included layers
will be printed recursively.
"""
function print_network(mdl; n=0, indent=0)

    top = indent == 0
    if top
        println("NNHelferlein neural network summary:")
        println(summary(mdl))
        println("Details:")
    else
        println(summary(mdl, indent=indent))
    end

    indent += 4
    println(" ")
    for pn in propertynames(mdl)
        p = getproperty(mdl, pn)

        if pn == :layers
            for l in p
                if l isa AbstractChain
                    n = print_network(l, n=n, indent=indent)
                    println(" ")
                elseif l isa AbstractLayer
                    println(summary(l, indent=indent))
                    n += 1
                else
                   print_summary_line(indent, "custom function", get_n_params(l)) 
                end
            end
        elseif p isa AbstractChain
            n = print_network(p, n=n, indent=indent)
            println(" ")
        elseif p isa AbstractLayer
            println(summary(p, indent=indent))
            n += 1
        elseif p isa AbstractArray
            for l in p
                n = print_network(l, n=n, indent=indent)
                #println(summary(l, indent=indent))
                n += 1
            end
        else
            print_summary_line(indent, "custom function", get_n_params(p)) 
        end
    end
    
        if top
            println(" ")
            println("Total number of layers: $n")
            println("Total number of parameters: $(get_n_params(mdl))")
        end
    return n
end

function print_summary_line(indent, line, params)

    LIN_LEN = 60
    s1 = " "^indent * line
    len = length(s1)
    gap = " "
    if len < LIN_LEN
        gap = " "^(LIN_LEN-len)
    end

    s2 = @sprintf("%8d params", params)

    return "$s1 $gap $s2"
end



"""
    struct VAE

Type for a generic variational autoencoder.

### Constructor:
    VAE(encoder, decoder)
Separate predefind chains (ideally, but not necessarily of type `Chain`) 
for encoder and decoder must be specified.
The VAE needs the 2 parameters mean and variance to define the distribution of each
code-neuron in the bottleneck-layer. In consequence the encoder output must be 2 times 
the size of the decoder input
(in case of dense layers: if encoder output is a 8-value vector,
4 codes are defined and the decoder input is a 4-value vector;
in case of convolutional layers the number of encoder output channels
must be 2 times the number of the encoder input channels - see the examples). 

### Signatures: 
    (vae::VAE)(x)
    (vae::VAE)(x,y)
Called with one argument, predict will be executed; 
with two arguments (args x and y should be identical for the autoencoder)
the loss will be returned.    

### Details:
The loss is calculated as the sum of element-wise error squares plus
the *Kullback-Leibler-Divergence* to adapt the distributions of the
bottleneck codes:
```math
\\mathcal{L} = \\frac{1}{2} \\sum_{i=1}^{n_{outputs}} (t_{i}-o_{i})^{2} - 
               \\frac{1}{2} \\sum_{j=1}^{n_{codes}}(1 + ln\\sigma_{c_j}^{2}-\\mu_{c_j}^{2}-\\sigma_{c_j}^{2}) 
```

Output
of the autoencoder is cropped to the size of input before
loss calculation (and before prediction); i.e. the output has always the same dimensions
as the input, even if the last layer generates a bigger shape.
"""
struct VAE <: AbstractNN
    layers
    VAE(layers::Vector) = new(layers)
    VAE(e,d) = new([e,d])
end

function (vae::VAE)(x, y=nothing)
    
    # encode and
    # calc size of decoder input (1/2 of encoder output):
    #
    size_in = size(x)
    x = vae.layers[1](x)
    size_dec_in = [size(x)...]
    size_dec_in[end-1] = size_dec_in[end-1] ÷ 2
    
    # separate μ and σ:
    #
    x = mat(x)
    code_size = size(x)
    n_codes = code_size[1] ÷ 2

    μ = x[1:n_codes,:]
    logσ² = x[n_codes+1:end,:]
    σ² = exp.(logσ²)
    σ = sqrt.(σ²)
    
    # variate:
    #
    ζ = oftype(μ, randn(Float32, size(μ)))
    
    x = μ .+ ζ .* σ
    
    # reshape codes to fit encoder input
    # and decode:
    #
    x = reshape(x, size_dec_in...)
    x = vae.layers[2](x)
    x = crop_array(x, size_in)
       
    # calc loss, if y given:
    #
    if isnothing(y)
        return x
    else
        loss = sum(abs2, x .- y) / 2
        loss_KL = -sum(1 .+ logσ² .- abs2.(μ) .- σ²) / 2
        return loss + loss_KL
    end
end
