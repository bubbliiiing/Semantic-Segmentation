from nets.unet import mobilenet_unet
model = mobilenet_unet(2,416,416)
model.summary()