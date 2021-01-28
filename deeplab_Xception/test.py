#---------------------------------------------#
#   该部分用于查看网络结构
#---------------------------------------------#
from nets.deeplab import Deeplabv3

if __name__ == "__main__":
    model = Deeplabv3(classes=2,OS=16)
    model.summary()
    for i in range(len(model.layers)):
        print(i,model.layers[i].name)
