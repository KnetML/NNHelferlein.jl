
"""
    function preproc_imagenet_vgg(img)
    function preproc_imagenet_resnet(img)
    function preproc_imagenet_resnetv2(img)

Image preprocessing for pre-trained ImageNet examples.
Preprocessing includes
+ bring RGB colour values into a range 0-255
+ standardise of colour values by substracting mean colour values
    (103.939, 116.779, 123.68) from RGB
+ changing colour channel sequence from RGB to BGR
+ normalising or scaling colour values.

Resize is **not** done, because this may be part of the
augmentation pipeline.

### Details
Unfortunately image preprocessing is not consistent between all 
pretrained Tenrflow/Keras applications.
As a result different preprocessing functions must beused for different 
pretrained applications:

+ **VGG16, VGG19**: `preproc_imagenet_vgg` 
  (colour space: BGR, values: 0 - 255, centered according to the
  imagenet training set)
+ **RESNET**: `preproc_imagenet_resnet` (identical to vgg)
+ **RESNET V2**: `preproc_imagenet_resnetv2` (colour space: BGR, 
  values: -1.0 - 1.0, scaled for each sample individually) 


### Examples:
The function can be used with the image loader; for prediction
with a trained model as:
```julia
pipl = CropRatio(ratio=1.0) |> Resize(224,224)
images = mk_image_minibatch("./example_pics", 16;
                    shuffle=false, train=false,
                    aug_pipl=pipl,
                    pre_proc=preproc_imagenet_vgg)
```

And for training something like:
```julia
pipl = Either(1=>FlipX(), 1=>FlipY(), 2=>NoOp()) |>
       Rotate(-5:5) |>
       ShearX(-5:5) * ShearY(-5:5) |>
       RCropSize(224,224)

dtrn, dvld = mk_image_minibatch("./example_pics", 16;
                    split=true, fr=0.2, balanced=false,
                    shuffle=true, train=true,
                    aug_pipl=pipl,
                    pre_proc=preproc_imagenet_vgg)
```
"""
function preproc_imagenet_vgg(img)

    return img |> imagenet_scale_255 |> imagenet_center_colours |> imagenet_bgr
end

preproc_imagenet_resnet = preproc_imagenet_vgg

function preproc_imagenet_resnetv2(img)

    ma = maximum(img)
    mi = minimum(img)
    s = 1 / (ma - mi) * 2
    img .= (img .- mi) .* s .- 1
    return img
end


function imagenet_scale_255(img)

    if maximum(img) <= 1.0
        img = img .* 255.0
    end
    return img
end

function imagenet_center_colours(img)

    (b, g, r) = (103.939, 116.779, 123.68)

    img[:,:,1] .-= r
    img[:,:,2] .-= g
    img[:,:,3] .-= b
    return img
end

function imagenet_bgr(img)

    y = similar(img)
    y[:,:,1] .= img[:,:,3]
    y[:,:,2] .= img[:,:,2]
    y[:,:,3] .= img[:,:,1]
    return y
end


# function preproc_imagenet(img)
# 
# 
#     img = img .* 255.0
#     (b, g, r) = (103.939, 116.779, 123.68)
# 
#     y = zeros(Float32, size(img))
#     y[:,:,1] .= img[:,:,3] .- b
#     y[:,:,2] .= img[:,:,2] .- g
#     y[:,:,3] .= img[:,:,1] .- r
# 
#     return y
# end

# TODO: adapt to pre-trained nets!

"""
    function get_imagenet_classes()

Return a list of all 1000 ImageNet class labels.
"""
function get_imagenet_classes()

    IMAGENET_CLASSES = joinpath(NNHelferlein.DATA_DIR, "imagenet", "classes.txt")

    if isfile(IMAGENET_CLASSES)
        classes = readlines(IMAGENET_CLASSES)
    else
        println("File with ImageNet class labels not found at")
        println("$IMAGENET_CLASSES.")

        classes = repeat(["?"], 1000)
    end
    return classes
end




"""
    function predict_imagenet(mdl; data, top_n=5)

Predict the ImageNet-class of images from the
predefined list of class labels.
"""
function predict_imagenet(mdl; data, top_n=5)

    classes = get_imagenet_classes()
    return predict_top5(mdl; data=data, top_n=top_n, classes=classes)
end
