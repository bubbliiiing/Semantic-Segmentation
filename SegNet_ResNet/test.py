#---------------------------------------------#
#   该部分用于查看网络结构
#---------------------------------------------#
from nets.segnet import resnet50_segnet

if __name__ == "__main__":
    model = resnet50_segnet(n_classes=2, input_height=416, input_width=416)
    model.summary()
