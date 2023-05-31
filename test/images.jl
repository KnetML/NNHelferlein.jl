# using MLDatasets: MNIST
using Base.Iterators
import Pkg; Pkg.add("Augmentor"); using Augmentor

function test_image_loader()
    augm = CropSize(28,28)
    trn, vld = mk_image_minibatch(joinpath("data", "flowers"),
                4; split=true, at=0.8,
                balanced=true, shuffle=true,
                train=true,
                aug_pipl=augm, pre_proc=preproc_imagenet_vgg)

    lab = get_class_labels(trn)

    return size(first(trn)[1]) == (28,28,3,4) &&
           size(lab) == (3,)
end


function test_image_preload()
    tst = mk_image_minibatch(joinpath("data", "flowers"),
                4; split=false, at=0.8,
                balanced=false, shuffle=false,
                train=false, pre_load=true,
                aug_pipl=nothing, pre_proc=nothing)

    return size(first(tst)) == (256, 256, 3, 4)
end


function test_image2arr()
        img = load(joinpath("data", "elecat", "cat.jpg"))
        x = image2array(img)
        return size(x) == (1024, 1001, 3) && eltype(x) == Float32
end

function test_array2image()
        img1 = array2image(rand(256,256,3))
        img2 = array2image(rand(256,256))
        img3 = array2RGB(rand(256,256))

        return eltype(img1) <: RGB &&
               eltype(img2) <: Gray &&
               eltype(img3) <: RGB
end


# imagenet:
#
function test_preproc_imagenet()

    img = rand(32,32,3)
    img = preproc_imagenet_vgg(img)
    img = preproc_imagenet_resnet(img)
    img = preproc_imagenet_resnetv2(img)
    return size(img) == (32,32,3)
end

function test_in_classes()
    c = get_imagenet_classes()
    return length(c) == 1000
end
