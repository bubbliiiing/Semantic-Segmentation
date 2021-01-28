#---------------------------------------------#
#   该部分用于查看网络结构
#---------------------------------------------#
from nets.pspnet import mobilenet_pspnet

if __name__ == "__main__":
    model = mobilenet_pspnet(2,576,576)
    model.summary()