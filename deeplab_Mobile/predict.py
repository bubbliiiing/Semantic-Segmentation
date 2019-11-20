from nets.deeplab import Deeplabv3
from PIL import Image
import numpy as np
import random
import copy
import os


class_colors = [[0,0,0],[0,255,0]]
NCLASSES = 2
HEIGHT = 416
WIDTH = 416


model = model = Deeplabv3(classes=2,input_shape=(HEIGHT,WIDTH,3))
model.load_weights("logs/ep015-loss0.052-val_loss0.100.h5")
imgs = os.listdir("./img")

for jpg in imgs:

    img = Image.open("./img/"+jpg)
    old_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    img = img.resize((WIDTH,HEIGHT))
    img = np.array(img)
    img = img/255
    img = img.reshape(-1,HEIGHT,WIDTH,3)
    pr = model.predict(img)[0]

    pr = pr.reshape((int(HEIGHT), int(WIDTH),NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((int(HEIGHT), int(WIDTH),3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))

    image = Image.blend(old_img,seg_img,0.3)
    image.save("./img_out/"+jpg)


