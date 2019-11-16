from nets.segnet import convnet_segnet

model = convnet_segnet(2,input_height=416,input_width=416)
model.summary()