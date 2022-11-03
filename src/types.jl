"""
    abstract type AbstractLayer
    abstract type Layer 

Mother type for layers hierarchy.
(The type `Layer` is kept for backward compatibility)
"""
abstract type AbstractLayer
end
abstract type Layer
end


""" 
    abstract type RecurrentUnit end

Supertype for all recurrent unit types.
Self-defined recurrent units which are a child of `RecurrentUnit`
can be used inside the 'Recurrent' layer.

### Interface
All subtypes of `RecurrentUnit` must provide the followning:
+ a constructor with signature `Type(n_inputs, n_units; kwargs)` and
    arbitrary keyword arguments.
+ an implementation of signature `(o::Recurrent)(x)`
    where `x` is a 3d- or 2d-array of shape [fan-in, mb-size, 1] or 
    [fan-in, mb-size].
    The function must return the result of one forward 
    computation for one step and return the hidden state
    and set the internal fields `h` and optionally `c`.
+ a field `h` (to store the last hidden state)
+ an optional field `c`, if the cell state is to be stored
    such as in a lstm unit.
"""
abstract type RecurrentUnit end




"""
    abstract type DataLoader

Mother type for minibatch iterators.
"""
abstract type DataLoader end

"""
    struct SequenceData <: DataLoader

Type for a generic minibatch iterator.
All NNHelferlein models accept minibatches of type `DataLoader`.

### Constructors:

    SequenceData(x; shuffle=true)

+ `x`: List, Array or other iterable object with the minibatches
+ `shuffle`: if `true`, minibatches are shuffled every epoch.
"""
mutable struct SequenceData <: DataLoader
    mbs
    l
    indices
    shuffle
    SequenceData(x; shuffle=true) = new(x, length(x), collect(1:length(x)), shuffle)
end

function Base.iterate(it::SequenceData, state=0)

    # shuffle if first call:
    #
    if it.shuffle && state == 0
        it.indices = Random.randperm(it.l)
    end
        
    if state >= it.l
        return nothing
    else
        state += 1
        return it.mbs[it.indices[state]], state
    end
end

Base.length(it::SequenceData) = it.l
Base.eltype(it::SequenceData) = eltype(first(it.mbs))

