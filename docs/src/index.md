# NNHelferlein - ADo's Neural Networks Little Helpers

The package provides helpers and utilities mainly to be
used with the Knet package to build artificial neural networks.
The German word *Helferlein* means something like *little helper*;
please pronounce it like `hell-fur-line`.

The package follows mainly the Knet-style; i.e. all networks can be
trained with the Knet-iterators, all layers can be used together with
Knet-style *quickly-self-written* layers, all Knet-networks can be trained
with tb_train(), all data providers can be used together, ...

The project is hosted here:    
<https://github.com/KnetML/NNHelferlein.jl>


## Installation

*NNHelferlein* is a registered package and 
 can be installed with the package manager as:

```JuliaREPL
] add NNHelferlein
```
or
```JuliaREPL
using Pkg
Pkg.add("NNHelferlein")
```


## First Steps

NNHelferlein provides quick and easy definition, training and
validation of neural network chains.

### Symbolic API
The Keras-like symbolic API allows for building simple Chains,
Classifiers and Regressors from predefined or self-written 
layers or functions.

A first example may be the famous MNIST handwriting recognition 
data. Let us assume the data is already loaded in minibatches 
in a `dtrn` iterator and a MLP shall do the job. 
The remainder is as little as:

```julia
mlp = Classifier(Dense(784, 256),
                 Dense(256, 64), 
                 Dense(64, 10, actf=identity)))


mlp = tb_train!(mlp, Adam, dtrn, epochs=10, split=0.8,
                acc_fun=accuracy, eval_size=0.2)
```

Chains may be built of type `Chain`, `Classifier` or `Regressor`.
Simple `Chain`s bring only a signature `model(x)` to compute 
forward computations
of a data-sample, a minibatch of data as well as many minibatches of data
(the dataset -here: dtrn- must be an iterable object that provides
one minibatch on every call).

`Classifier`s and `Regressor`s in addition already come with signatures
for loss calculation of (x,y)-minibatches (`model(x,y)`) 
with crossentropy loss
(i.e. negative log-likelihood) and square-loss respectively. This is why 
for both types the last layer must not have an activation function
(the *Helferlein* `Dense`-layer comes with a logistic/sigmoid activation
by default; alternatively the `Linear`-layer can be used that have 
no default activation function).

The function `tb_train!()`
updates the model with the possibility to specify optimiser, training
and validation data or an optional split ratio to perform a random 
training/validation split. The function offers a multitude of 
other options (see the API-documentation for details) and writes
tensorboard log-files that allow for online monitoring of the 
training progress during training via tensorboard.

A second way to define a model is the `add_layer!()`-syntax, here shown
for a simple LeNet-like model for the same problem:

```julia
lenet = Classifier()

add_layer!(lenet, Conv(5,5,1,20))
add_layer!(lenet, Pool())
add_layer!(lenet, Conv(5,5,20,50))
add_layer!(lenet, Pool())
add_layer!(lenet, Flat())
add_layer!(lenet, Dense(800,512))
add_layer!(lenet, Dense(512,10, actf=identity))

mlp = tb_train!(lenet, Adam, dtrn, epochs=10, split=0.8,
                acc_fun=accuracy, eval_size=0.2)
```

As an alternative the `+`-operator is overloaded to be able to 
just add layers to a network:

```julia
julia> mdl = Classifier() + Dense(2,5)
julia> mdl = mdl + Dense(5,5) + Dense(5,1, actf=identity)
julia> summary(mdl)

NNHelferlein neural network summary:
Classifier with 3 layers,                                           51 params
Details:
 
    Dense layer 2 → 5 with sigm,                                    15 params
    Dense layer 5 → 5 with sigm,                                    30 params
    Dense layer 5 → 1 with identity,                                 6 params
 
Total number of layers: 3
Total number of parameters: 51
```


Of course, all possibilities can be combined as desired; the
following code gives a similar model:

```julia
filters = Chain(Conv(5,5,1,20),
                Pool(),
                Conv(5,5,20,50),
                Pool())
classif = Chain(Dense(800,512),
                Dense(512,10, actf=identity))

lenet2 = Classifier(filters, 
                   Flat())
add_layer!(lenet2, classif)

mlp = tb_train!(lenet2, Adam, dtrn, epochs=10, split=0.8,
                acc_fun=accuracy, eval_size=0.2)
```

Models can be summarised with `summary()` or the `print_network()`-helper:

```julia
julia> summary(lenet)
Neural network summary:
Classifier with 7 layers,                                       440812 params
Details:
 
    Conv layer 1 → 20 (5,5) with relu,                             520 params
    Pool layer,                                                      0 params
    Conv layer 20 → 50 (5,5) with relu,                          25050 params
    Pool layer,                                                      0 params
    Flat layer,                                                      0 params
    Dense layer 800 → 512 with sigm,                            410112 params
    Dense layer 512 → 10 with identity,                           5130 params
 
Total number of layers: 7
Total number of parameters: 440812
```

### Free model definition
Another way of model definition gives the full freedom 
to define a forward function as pure Julia code. 
In the Python world this type of definition is often referred to  
as the functional API - in the Julia world we hesitate calling 
it an API, 
because at the end of the day all is just out-of-the-box Julia!
Each model just needs a type, able to store all parameters, 
a signature `model(x)` to compute a forward run and predict
the result and a signature `model(x,y)` to calculate the loss.

For the predefined `Classifier` and `Regressor` types the signatures are 
predefined - for own models, this can be easily done in a few lines of
code.

The LeNet-like example network for MNIST may be written as:

#### The type and constructor:
```julia
struct LeNet
    drop1
    conv1
    pool1
    conv2
    pool2
    flat
    drop2
    mlp
    predict
    function LeNet(;drop=0.2)
        return new(Dropout(drop),
                   Conv(5,5,1,20),
                   Pool(),
                   Conv(5,5,20,50),
                   Pool(),
                   Flatten(),
                   Dropout(drop)
                   Dense(800, 512),
                   Dense(512, 10, actf=identity))
end
```
Of course the model may be configured by giving the constructor
more parameters.
Also the code may be written better organised by combining
layers to `Chains`.


#### The forward signature:

Brute-force definition:

```julia
function (nn::LeNet)(x)
    x = nn.drop1(x)
    x = nn.conv1(x)
    x = nn.pool1(x)
    x = nn.conv2(x)
    x = nn.pool2(x)
    x = nn.flat(x)
    x = nn.drop2(x)
    x = nn.mlp(x)
    x = nn.predict(x)
    return x
end
```

... or a little bit more elegant:

```julia

function (nn::LeNet)(x)
    layers = (nn.drop1, nn.conv1, nn.pool1, 
              nn.conv2, nn.pool2, nn.flat, 
              nn.drop2, nn.mlp, nn.predict)

    for layer in layers
        x = layer(x)
    end
    return x
end
```

... or a little bit more elegant:

```julia
function (nn::LeNet)(x)
    layers = (nn.drop1, nn.conv1, nn.pool1, 
              nn.conv2, nn.pool2, nn.flat, 
              nn.drop2, nn.mlp, nn.predict)

    return foldl((x,layer)->layer(x), layers, init=x)
end
```

... or a little more structured:

```julia
function (nn::LeNet)(x)
    filters = Chain(nn.drop1, nn.conv1, nn.pool1, 
              nn.conv2, nn.pool2)
    classifier = Chain(nn.drop2, nn.mlp, nn.predict)

    x = filters(x)
    x = nn.flat(x)
    x = classifier(x) 
    return x
end
```

#### The loss-signature:
```julia
function (nn::LeNet)(x,y)
    return nll(nn(x), y)
end
```

Here we use the `Knet.nll()` function to calculate the crossentropy. 

That's it!
Every object of type `LeNet` is now a fully functional model,
which can be trained with `tb_train!()`.

Belive it or not - that's all you need to leave the 
limitations of the Python world behind and playfully design any 
innovative neural network in just a couple of lines of Julia code.

The next step is to have a look at the examples
in the GitHub repo:

```@contents
Pages = [
    "examples.md"
    ]
Depth = 2
```

## Overview

```@contents
Pages = [
    "overview.md"
    ]
Depth = 2
```
## Datasets

Some datasets as playground-data are provided with the
package. Maybe more will follow...

+ *MIT Normal Sinus Rhythm Database* is a modified version of the 
  Physionet dataset, adapted for use in machine leraning
  (see the docstring of `dataset_mit_nsr()` for details).

+ the famous *MNIST* dataset.

+ R.A. Fisher's *Iris* dataset.

## API Reference

```@contents
Pages = [
    "api.md"
    ]
Depth = 2
```

## Index

```@index
```

## Changelog

The history can be found here: [ChangeLog of NNHelferlein package](@ref)

