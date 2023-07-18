# funs for minibatch manipulation
#

"""
    function split_minibatches(it, at=0.8; shuffle=true)

Return 2 iterators of type `PartialIterator` which iterate only parts of the 
states of the iterator `it`. 
Be aware that the partial iterators will not contain copies of the data
but instead forward the data provided by the iterator `it`.

The function can be used to split an iterator of minibatches into train- 
and validation iterators, without copying any data.
As the `PartialIterator` objects work with the states of the inner iterator,
it is important *not* to shuffle the inner iterator (in this case the 
composition of the partial iterators would change and training and validation data 
may be mixed!).

### Arguments:
+ `it`: Iterator to be splitted. The list of allowed states is created by
        performing a full iteration once.
+ `at`: Split point. The first returned iterator will include the given 
        fraction (default: 80%) of the states.
+ `shuffle`: If true, the elements are shuffled at each restart of the iterator.
"""
function split_minibatches(it, at=0.8; shuffle=true)
    
    # make sure, the inner iterator is not shuffled:
    #
    if hasproperty(it, :shuffle) && it.shuffle
        println("Warning: shuffle=true not allowed for the iterator to be splitted!")
        println("Instead, training and validation minibatches will be shuffled seperately.")

        it.shuffle = false
        shuffle = true
    end


    # collect all valid states of it
    # ann nothing for the first state and remove last state 
    # (that delivers nothing)
    #
    states = []
    push!(states, nothing)
    e = iterate(it)
    while !isnothing(e)
        state = e[2]
        push!(states, state)
        e = iterate(it, state)
    end
    pop!(states)
    
    # shuffle indices if demanded:
    #
    if shuffle
        Random.shuffle!(states)
    end
    
    # create index lists for trn and vld:
    #
    n_trn = Int(round(length(states) * at))
        
    if n_trn == 0
        n_trn = 1
        #trn_idx = []
        #vld_idx = states
    elseif n_trn == length(states)
        n_trn = length(states) - 1
        #trn_idx = states
        #vld_idx = []
    end
    trn_idx = states[1:n_trn]
    vld_idx = states[n_trn+1:end]
    
    return PartialIterator(it, trn_idx, shuffle=shuffle), PartialIterator(it, vld_idx, shuffle=shuffle) 
end




"""
    struct PartialIterator <: DataLoader

The `PartialIterator` wraps any iterator and will only iterate the states
specified in the list `indices`. 

### Constuctors

    PartialIterator(inner, indices; shuffle=true) 

Type of the states must match
the states of the wrapped iterator `inner`. A `nothing` element may be 
given to specify the first iterator element.

If `shuffle==true`, the list of indices are shuffled every time the
`PartialIterator` is started.
"""
mutable struct PartialIterator <: DataLoader
    inner
    indices
    l
    shuffle
    PartialIterator(inner, indices; shuffle=true) = new(inner, indices, length(indices), shuffle)
end

function Base.iterate(it::PartialIterator, state=0)
    
    if it.shuffle && state == 0
        Random.shuffle!(it.indices)
    end
    
    if state >= it.l
        return nothing
    else
        state += 1
        inner_state = it.indices[state]
        
        if isnothing(inner_state)
            return iterate(it.inner,)[1], state
        else
            return iterate(it.inner, inner_state)[1], state
        end
    end
end

Base.length(it::PartialIterator) = it.l
Base.eltype(it::PartialIterator) = eltype(first(it.inner))


""" 
    type MBNoiser

Iterator to wrap any Knet.Data iterator of minibatches in 
order to add random noise.    
Each value will be multiplied with a random value form 
Gaussian noise with mean=1.0 and sd=σ.

### Construtors:
    MBNoiser(mbs::Knet.Data, σ)
    MBNoiser(mbs::Knet.Data; σ=0.01)

+ `mbs`: iterator with minibatches
+ `σ`: standard deviation for the Gaussian noise

### Example:
```juliaREPL
julia> trn = minibatch(x)
julia> tb_train!(mdl, Adam, MBNoiser(trn, σ=0.1))
julia> mbs_noised = MBNoiser(mbs, 0.05)
```
"""
struct MBNoiser  <: DataLoader
    mbs
    σ
    MBNoiser(mbs, sd=0.01; σ=sd) = new(mbs, σ)
end

# TODO: size on-the-fly


# first call:
#
function Base.iterate(nr::MBNoiser) 
    return iterate(nr,0)
end

# subsequent calls with state:
#
function Base.iterate(nr::MBNoiser, state)
    next_inner = iterate(nr.mbs, state)
    if isnothing(next_inner)
        return nothing
    else
        next_mb, next_state = next_inner
        return (do_noise(next_mb[1], nr.σ) , next_mb[2]), next_state
    end
end

# and length = length of inner iterator:
#
Base.length(it::MBNoiser) = length(it.mbs)


# not exposed inner funs:
#
function do_noise(x, σ)
    x = x .* ifgpu( randn(Float32, size(x)...) .* σ .+ 1 )
    return(x)
end





"""
    struct MBMasquerade  <: DataLoader

Iterator wrapper to partially mask training data of a minibatch 
iterator of type `Knet.Data` or `NNHelferlein.DataLoader`.

### Constructors:
    MBMasquerade(it, rho=0.1; mode=:noise, value=0)
    MBMasquerade(it; ρ=0.1, mode=:noise, value=0)

The constructor may be called with the density `rho` as normal
argument or `ρ` as keyword argument.

### Arguments:
+ `it`: Minibatch iterator that must deliver (x,y)-tuples of 
        minibatches
+ `ρ=0.1` or `rho`: Density of mask; a value of 1.0 will mask everything,
        a value of 0.0 nothing.
+ `value=0`: the value with which the masking is done.
+ `mode=:noise`: type of masking (only `:noise` implemented yet):
    + `:noise`: randomly distributed single values of the 
            training data will be overwitten with `value`.

### Examples:

```juliaREPL
julia> dtrn 
26-element Knet.Train20.Data{Tuple{CuArray{Float32}, Array{UInt8}}}

julia> mtrn = Masquerade(dtrn, 0.5, value=2.0h)
Masquerade(26-element Knet.Train20.Data{Tuple{CuArray{Float32}, Array{UInt8}}}, 0.5, 2.0, :noise)
```
"""
struct MBMasquerade  <: DataLoader
    it
    ρ
    value
    mode
    MBMasquerade(it, rho=0.1; ρ=rho, mode=:noise, value=0) = 
        new(it, ρ, Float32(value), mode)
end

function Base.iterate(it::MBMasquerade)
    return iterate(it, 0)
end

function Base.iterate(it::MBMasquerade, state)
    
    next_inner = iterate(it.it, state)
    if isnothing(next_inner)
        return nothing
    end
    
    (x,y), next_state = next_inner
    
    if it.mode == :noise
        x = do_mask(x, it.ρ, it.value)
    #elseif it.mode == :patch
    #    x = do_patch(x, it.ρ, it.value)
    end
    
    return (x,y), next_state
end

Base.length(it::MBMasquerade) = length(it.it)



# not exposed inner funs:
#
function do_mask(x, ρ, value)
    
    mask = rand(size(x)...) .< ρ
    mask = ones(eltype(x), size(x)) .* mask       # make bool to numeric
    mask = typeof(x)(mask)                         # make syme type as x
    value = eltype(x)(value)
    x = x .- value                                  # set msaked positions to value
    x = x .* mask
    x = x .+ value
    return x
end

function do_patch(x, ρ, value)

    # last dim is always minibatch:
    # apply to one dim data:
    #
    if ndims(x) == 2
        p_size = [size(x,1) .* ρ .|> round .|> Int, size(x,2)]
    
    # apply to 2 and more-dim data# patch is always on the first
    # 2 dims:
    #
    else
        p_size = [size(x,1), size(x,2)] .* √ρ .|> round .|> Int
        if ndims(x) > 2
            p_size = vcat(p_size, [size(x)...][3:end])
        end
    end
    p_size[p_size.==0] .= 1 

    p_start = [rand(collect(1:s-p+1)) for (s,p) in zip(size(x), p_size) ]
    p_end = p_start .+ p_size .- 1
    ranges = ((i:j) for (i,j) in zip(p_start, p_end))

    x[ranges...] .= value
    return x
end
