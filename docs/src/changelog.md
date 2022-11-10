# ChangeLog of NNHelferlein package

+ pretrained fixed
+ ResNetBlock added
+ ResNet added
+ Padding layer added
+ print_network changed to summary
+ Pretrained nets saved at zenodo and simplified constructors added
+ AbstractNN and AbstractLayer added
+ copy model and save/load as JLD2 added

### v1.1.2
+ Depthwise conv-layer added (experimental)
+ focal loss functions added to classifier 
+ FeatureSelection layer added
+ explicit signature added for 3d-convolution
+ train: possibility to disable tensorboard logs
+ train: possibility to return losses and accs for 
  plotting after training

### v1.1.1
+ some docstring cosmetics
+ Activation Layers added
+ layer GlobalAveragePoling added
+ pre-trained vgg example fixed for changed "import-HDF"-interface
+ hdf5 import with all kwargs possible
+ added: Layer + Layer = Chain
+ changelog added to docu

### v1.1.0
+ documentation for release added
+ split_minibatches() made stable (never returns an empty iterator)
+ docs slightly re-organised
+ Gaussian Layer added
+ minibatch iterator for masking added


### v1.0.0
+ initial release

