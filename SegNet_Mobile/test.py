from nets.segnet import mobilenet_segnet

model = mobilenet_segnet(2,input_height=416,input_width=416)
model.summary()