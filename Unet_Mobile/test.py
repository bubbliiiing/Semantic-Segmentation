#---------------------------------------------#
#   该部分用于查看网络结构
#---------------------------------------------#
from nets.unet import mobilenet_unet

if __name__ == "__main__":
    model = mobilenet_unet(2, 416, 416)
    model.summary()