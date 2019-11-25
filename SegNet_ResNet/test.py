from nets.segnet import resnet50_segnet
model = resnet50_segnet(n_classes=2,input_height=416, input_width=416)
model.summary()

for i,layer in enumerate(model.layers):
    print(i,layer.name)