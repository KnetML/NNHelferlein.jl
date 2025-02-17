
# Generate adapt_structure()-funktions for Adapt-Package
# for all Helferlein Types:
#
Adapt.@adapt_structure Param 
Adapt.@adapt_structure Recurrent
Adapt.@adapt_structure RNN
Adapt.@adapt_structure MultiHeadAttn
Adapt.@adapt_structure AttnBahdanau
Adapt.@adapt_structure AttnLuong
Adapt.@adapt_structure AttnDot
Adapt.@adapt_structure AttnLocation
Adapt.@adapt_structure AttnInFeed

# all layer types are handled by the Layer type:
#
#Adapt.@adapt_structure Dense
#Adapt.@adapt_structure Conv
#Adapt.@adapt_structure Classifier 
#Adapt.@adapt_structure Regressor 
#
#function Adapt.adapt_structure(to, x::Vector{Any})
#    return [adapt(to, elem) for elem in x]
#end
#
# adapt for all Layers:
#
function Adapt.adapt_structure(to::Type, x::Union{AbstractNN, AbstractChain, AbstractLayer})
    

    function adapt_property(to, x, pn)
        
        p = getfield(x, pn)
        
        println("adapt_property: ", pn)
        println("type: $(typeof(p))")
        println("$(summary(p))")
        println(" ")

        if p isa Tuple
            return Tuple([Adapt.adapt(to, elem) for elem in p])
        elseif p isa AbstractArray
            return Array([Adapt.adapt(to, elem) for elem in p])
        elseif p isa Base.Pairs
            return Adapt.adapt(to, p)
            # for (k, v) in p
            #     println("key: $k, value: $v")
            # end
            # ks = keys(p)
            # vs = values(p)
            # nvs = [Adapt.adapt(to, elem) for elem in vs]
            # return typeof(p)((nvs), (ks))
        else
            return Adapt.adapt(to, getfield(x, pn))
        end
    end

    println(">> adapt_structure, Layer: $(summary(x)) \n")
    T = typeof(x)
    new_fields = [adapt_property(to, x, pn) for pn in propertynames(x)]
    return T(new_fields...)
end


# adapt for all Nets and Chains:
#
# function Adapt.adapt_structure(to, x::AbstractNN)
#     T = typeof(x)
#     if hasproperty(x, :layers)
#         layers = Any[Adapt.adapt(to, l) for l in x.layers]
#         if hasproperty(x, :loss)
#             return T(layers, x.loss)
#         else
#             return T(layers)
#         end
#     else
#         new_fields = [Adapt.adapt(to, getfield(x, pn)) for pn in propertynames(x)]
#         return(T(new_fields...))
#     end
# end

function Adapt.adapt_structure(to::Type, x::NNHelferlein.Conv)

    wn = Adapt.adapt(to, x.w)
    bn = Adapt.adapt(to, x.b)
    return Conv(wn, bn, x.actf; x.kwargs... )
end


"""
    copy_network(mdl::AbstractNN; to=:gpu)

Returns a copy of a Helferlein model.
*cave: the copy is generated by `Adapt.adapt()` and no
deep copy!*

### Arguments:
+ `mdl`: Network model of type `AbstractNN`.
+ `to=:gpu`: by default all parameters of the copy are `CuArrays` for
             GPU usage. If `to=:cpu` is specified, parameters 
             are Arrays and the model will be processed in the cpu.
"""
function copy_network(mdl::AbstractNN; to=:gpu)

    if to == :gpu && CUDA.functional()
        return Adapt.adapt(CuArray, mdl)
    elseif to == :cpu
        return Adapt.adapt(Array, mdl)
    else
        p = first(params(mdl))
        return Adapt.adapt(typeof(p.value), mdl)
    end
end


"""
    save_network(fname, mdl)

Save a model as jld2-file.

### Arguments:
+ `fname`: filename; if the name does not end with the extension `.jld2`, 
          it will be added.
+ `mdl`: network model to be saved. The model will be copied to a 
          cpu-based model via `copy_network(mdl, to=:cpu)` before
          saving, to remove hardware dependencies of 
          parameters on the gpu.
"""
function save_network(fname, mdl)
    
    if !endswith(fname, ".jld2")
        fname = fname * ".jld2"
    end

    JLD2.save(fname, Dict("helferlein"=>copy_network(mdl, to=:cpu)))
end

"""
    load_network(fname; to=:gpu)

Load a model from a jld2-file.

### Arguments:
+ `fname`: filename; if the name does not end with the extension `.jld2`, 
           it will be added.
+ `to=:gpu`: by default, parameters are loaded as CuArrays, if
           a functional gpu is detected. If `to=:cpu` is specified
           parameters are loaded as cpu-arrays.
"""
function load_network(fname; to=:gpu)

    if !endswith(fname, ".jld2")
        fname = fname * ".jld2"
    end

    mdl = JLD2.load(fname, "helferlein")
    return copy_network(mdl, to=to)
end
