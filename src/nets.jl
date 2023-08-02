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
    struct Regressor <: AbstractNN

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
    function add_layer!(n::Union{NNHelferlein.AbstractNN, NNHelferlein.AbstractChain}, l)

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




function summary_scan_properties(mdl; n=0, indent=0)

    for pn in propertynames(mdl)
        p = getproperty(mdl, pn)

        if p isa AbstractChain || p isa AbstractLayer 
            n += summary(p, indent=indent)
        elseif pn == :layers || p isa AbstractArray
            for l in p
                if l isa AbstractChain || l isa AbstractLayer 
                    n += summary(l, indent=indent)
                else
                    n += any_summary(l, indent=indent)
                end
            end
        end
    end
    return n
end


# simulate summary for unknown layer type:
#
function any_summary(layer; indent=indent)
    n = get_n_params(layer)
    s1 = "Function layer of type $(typeof(layer)),"
    println(print_summary_line(indent, s1, n))
    return 1
end




"""
    function summary(mdl)

Print a network summary of any model of Type `AbstractNN`, 
`AbstractChain` or `AbstractLayer`.
"""
function Base.summary(mdl::AbstractNN; n=0, indent=0)

    println("NNHelferlein neural network of type $(typeof(mdl)):")
    println(" ")

    indent += 2
    n += summary_scan_properties(mdl, n=0, indent=indent)

    if hasproperty(mdl, :loss)
        println(" ")
        println("  Loss function: $(mdl.loss)")
    end

    println(" ")
    println("Total number of layers: $n")
    println("Total number of parameters: $(get_n_params(mdl))")
    return n
end



function Base.summary(mdl::AbstractChain; n=0, indent=0)

    println(" "^indent*"Chain of type $(typeof(mdl)):")

    indent += 2
    n += summary_scan_properties(mdl, n=0, indent=indent)
end


function print_network(l::AbstractLayer, n=0, indent=0)

    println(summary(l, intent=indent))
    return 1
end




"""
    function print_network(mdl::AbstractNN)

Alias to `summary()`, kept for backward compatibility only.
"""
print_network(mdl; n=0, indent=0) = summary(mdl, n=n, indent=indent)


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
    struct VAE   <: AbstractNN

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

### KL-training parameters:
The parameter β is by default set to 1.0, i.e. mean-squared error and KL 
has the same weights. The functions `set_beta(vae, beta)` and
`get_beta(vae)` can be used to set and get the β used in training.
With β=0.0 no KL-loss will be used.
"""
struct VAE <: AbstractNN
    layers
    p    # dictionary of additional parameters
    VAE(layers::Vector; beta=1.0) = new(layers, Dict(:ramp=>1.0, :beta_max=>beta, :ramp_up=>false, :steps=>0, :delta=>0.0))
    VAE(e,d; beta=1.0) = new([e,d], Dict(:ramp=>1.0, :beta_max=>beta, :ramp_up=>false, :steps=>0, :delta=>0.0))
end

"""
    function get_beta(vae::VAE; ramp=false)

Return a `Dict` with the current VAE-parameters beta and ramp-up.

### Arguments:
+ `ramp=false`: if `true`, a vector of β for all ramp-up steps is returned.
                This way, the ramp-up phase can be visualised:
                <img src="./assets/vae-beta-range.png"/>
"""
function get_beta(vae::VAE; ramp=false)

    if ramp 
        β = []
        ramp = sigm(-10.0)
        for i = 1:vae.p[:steps]
            ramp = ramp + ramp*(1-ramp) * vae.p[:delta]
            push!(β, ramp * vae.p[:beta_max])
        end
        return β
    else
        return vae.p
    end
end




"""
   function set_beta!(vae::VAE, β_max; ramp_up=false, steps=0)

Helper to set the current value of the VAE-parameter beta
and ramp-up settings.

VAE loss is calculated as (mean of error squares) + β * (mean of KL divergence).

### Ramp-up:
In case of `ramp_up=true`, β starts with almost 0.0 (`sigm(-10.0)` ≈4.5e-5) and 
reaches almost 1.0 after `steps` steps, following a sigmoid curve.
`steps` should be more than 25, to avoid rounding errors in the calculation of
the derivative of the sigmoid function.
"""
function set_beta!(vae::VAE, β_max; ramp_up=false, steps=1)

    if ramp_up
        vae.p[:ramp] = sigm(-10.0)
        vae.p[:beta_max] = β_max
        vae.p[:ramp_up] = true
        vae.p[:steps] =steps
        vae.p[:delta] = 20/steps
        println("Weights β for KL-Loss set to ramp up from 0.0 to $β_max in $steps steps.")
    else
        vae.p[:ramp] = 1.0
        vae.p[:beta_max] = β_max
        vae.p[:ramp_up] = false
        vae.p[:steps] = 1
        vae.p[:delta] = 1.0

        println("Weights β for KL-Loss set to constant $β_max.")
    end
end

function (vae::VAE)(x::AbstractArray, y::AbstractArray) 
    
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
    loss = mean(abs2, x .- y) / 2 
    loss_KL = - mean(1 .+ logσ² .- abs2.(μ) .- σ²) / 2 

    # ramp-up beta:
    #
    if vae.p[:ramp_up]
        vae.p[:ramp] += vae.p[:ramp]*(1-vae.p[:ramp]) * vae.p[:delta]
    end
    return loss + vae.p[:ramp] * vae.p[:beta_max] * loss_KL
end

function (vae::VAE)(x::AbstractArray)
    
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
    # logσ² = x[n_codes+1:end,:]
    
    x = μ
    
    # reshape codes to fit encoder input
    # and decode:
    #
    x = reshape(x, size_dec_in...)
    x = vae.layers[2](x)
    x = crop_array(x, size_in)
       
    return x
end
