#---------------------------------------------#
#   该部分用于查看网络结构
#---------------------------------------------#
from nets.segnet import mobilenet_segnet

if __name__ == "__main__":
    model = mobilenet_segnet(2,input_height=416,input_width=416)
    model.summary()
