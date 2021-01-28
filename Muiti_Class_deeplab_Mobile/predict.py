import colorsys
import copy
import os
import random

import numpy as np
from PIL import Image

from nets.deeplab import Deeplabv3


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0,0,0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image,nw,nh

def get_class_colors(num_classes):
    if num_classes <= 21:
        colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 12)]
    else:
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(num_classes), 1., 1.)
                    for x in range(len(num_classes))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
    return colors

if __name__ == "__main__":
    #---------------------------------------------#
    #   定义输入图片的高和宽，以及种类数量
    #---------------------------------------------#
    HEIGHT = 512
    WIDTH = 512
    #---------------------------------------------#
    #   背景 + 需要去区分的类的个数
    #---------------------------------------------#
    NCLASSES = 21
    #---------------------------------------------------#
    #   根据种类的数量获取颜色
    #---------------------------------------------------#
    class_colors = get_class_colors(NCLASSES)

    #---------------------------------------------#
    #   载入模型
    #---------------------------------------------#
    model = Deeplabv3(classes=NCLASSES,input_shape=(HEIGHT,WIDTH,3))
    #--------------------------------------------------#
    #   载入权重，训练好的权重会保存在logs文件夹里面
    #   我们需要将对应的权重载入
    #   修改model_path，将其对应我们训练好的权重即可
    #   下面只是一个示例
    #--------------------------------------------------#
    model.load_weights("model_data/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5")

    #--------------------------------------------------#
    #   对imgs文件夹进行一个遍历
    #--------------------------------------------------#
    imgs = os.listdir("./img/")
    for jpg in imgs:
        #--------------------------------------------------#
        #   打开imgs文件夹里面的每一个图片
        #--------------------------------------------------#
        img = Image.open("./img/"+jpg)
        
        old_img = copy.deepcopy(img)
        orininal_h = np.array(img).shape[0]
        orininal_w = np.array(img).shape[1]

        #--------------------------------------------------#
        #   对输入进来的每一个图片进行letterbox_image
        #--------------------------------------------------#
        img, nw, nh = letterbox_image(img, [WIDTH, HEIGHT])
        img = np.array(img) / 127.5 - 1
        img = img.reshape(-1,HEIGHT,WIDTH,3)

        #--------------------------------------------------#
        #   将图像输入到网络当中进行预测
        #--------------------------------------------------#
        pr = model.predict(img)[0]
        pr = pr.reshape((HEIGHT,WIDTH,NCLASSES)).argmax(axis=-1)
        pr = pr[int((HEIGHT-nh)//2):int((HEIGHT-nh)//2+nh), int((WIDTH-nw)//2):int((WIDTH-nw)//2+nw)]

        #------------------------------------------------#
        #   创建一副新图，并根据每个像素点的种类赋予颜色
        #------------------------------------------------#
        seg_img = np.zeros((nh, nw,3))
        for c in range(NCLASSES):
            seg_img[:,:,0] += ( (pr[:,: ] == c )*( class_colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c )*( class_colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( class_colors[c][2] )).astype('uint8')

        seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
        #------------------------------------------------#
        #   将新图片和原图片混合
        #------------------------------------------------#
        image = Image.blend(old_img,seg_img,0.5)
        
        image.save("./img_out/"+jpg)


