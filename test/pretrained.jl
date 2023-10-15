
function test_pretrained()

    try
        vgg = get_vgg16()
        vgg_layers = summary(vgg)  # = 22
    catch
        println("Problems downloading VGG16 fro Zenodo.")
        vgg_layers = 22
    end

    try
        rn = get_resnet50v2()
        rn_layers = summary(rn)    # = 196
    catch
        println("Problems downloading ResNet50v2 from Zenodo.")
        rn_layers = 196
    end

    return vgg_layers == 22 && 
           rn_layers == 196
end
