
function test_pretrained()

    vgg = get_vgg16()
    rn = get_resnet50v2()

    vgg_layers = summary(vgg)  # != 22
    rn_layers = summary(rn)    # != 196

    return vgg_layers == 22 && 
           rn_layers == 196
end
