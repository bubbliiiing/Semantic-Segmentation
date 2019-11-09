from nets.pspnet import mobilenet_pspnet
model = mobilenet_pspnet(2,576,576)
model.summary()