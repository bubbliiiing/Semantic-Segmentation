#---------------------------------------------#
#   该部分用于查看网络结构
#---------------------------------------------#
from nets.deeplab import Deeplabv3

if __name__ == "__main__":
    deeplab_model = Deeplabv3()
    deeplab_model.summary()

    for i in range(len(deeplab_model.layers)):
        print(i,deeplab_model.layers[i].name)
