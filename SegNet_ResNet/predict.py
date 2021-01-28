import copy
import os
import random

import numpy as np
from PIL import Image

from nets.segnet import resnet50_segnet

if __name__ == "__main__":
    #---------------------------------------------------#
    #   定义了输入图片的颜色，当我们想要去区分两类的时候
    #   我们定义了两个颜色，分别用于背景和斑马线
    #   [0,0,0], [0,255,0]代表了颜色的RGB色彩
    #---------------------------------------------------#
    class_colors = [[0,0,0],[0,255,0]]
    #---------------------------------------------#
    #   定义输入图片的高和宽，以及种类数量
    #---------------------------------------------#
    HEIGHT = 416
    WIDTH = 416
    #---------------------------------------------#
    #   背景 + 斑马线 = 2
    #---------------------------------------------#
    NCLASSES = 2
    
    #---------------------------------------------#
    #   载入模型
    #---------------------------------------------#
    model = resnet50_segnet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
    #--------------------------------------------------#
    #   载入权重，训练好的权重会保存在logs文件夹里面
    #   我们需要将对应的权重载入
    #   修改model_path，将其对应我们训练好的权重即可
    #   下面只是一个示例
    #--------------------------------------------------#
    model.load_weights("logs/ep053-loss0.021-val_loss0.028.h5")

    #--------------------------------------------------#
    #   对imgs文件夹进行一个遍历
    #--------------------------------------------------#
    imgs = os.listdir("./img")
    for jpg in imgs:
        #--------------------------------------------------#
        #   打开imgs文件夹里面的每一个图片
        #--------------------------------------------------#
        img = Image.open("./img/"+jpg)
        
        old_img = copy.deepcopy(img)
        orininal_h = np.array(img).shape[0]
        orininal_w = np.array(img).shape[1]

        #--------------------------------------------------#
        #   对输入进来的每一个图片进行Resize
        #   resize成[HEIGHT, WIDTH, 3]
        #--------------------------------------------------#
        img = img.resize((WIDTH,HEIGHT), Image.BICUBIC)
        img = np.array(img) / 255
        img = img.reshape(-1, HEIGHT, WIDTH, 3)

        #--------------------------------------------------#
        #   将图像输入到网络当中进行预测
        #--------------------------------------------------#
        pr = model.predict(img)[0]
        pr = pr.reshape((int(HEIGHT/2), int(WIDTH/2), NCLASSES)).argmax(axis=-1)

        #------------------------------------------------#
        #   创建一副新图，并根据每个像素点的种类赋予颜色
        #------------------------------------------------#
        seg_img = np.zeros((int(HEIGHT/2), int(WIDTH/2),3))
        for c in range(NCLASSES):
            seg_img[:, :, 0] += ((pr[:,: ] == c) * class_colors[c][0]).astype('uint8')
            seg_img[:, :, 1] += ((pr[:,: ] == c) * class_colors[c][1]).astype('uint8')
            seg_img[:, :, 2] += ((pr[:,: ] == c) * class_colors[c][2]).astype('uint8')

        seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
        #------------------------------------------------#
        #   将新图片和原图片混合
        #------------------------------------------------#
        image = Image.blend(old_img,seg_img,0.3)
        
        image.save("./img_out/"+jpg)


