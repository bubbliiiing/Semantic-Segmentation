from nets.deeplab import Deeplabv3
from PIL import Image
import numpy as np
import random
import copy
import os

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

random.seed(0)
class_colors = [[random.randint(0,255),random.randint(0,255),random.randint(0,255)] for _ in range(21)]
NCLASSES = 21
HEIGHT = 512
WIDTH = 512
model = Deeplabv3(classes=21,input_shape=(HEIGHT,WIDTH,3))
model.load_weights("logs/last1.h5")

imgs = os.listdir("./img")

for jpg in imgs:

    img = Image.open("./img/"+jpg)
    old_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    img,nw,nh = letterbox_image(img,[HEIGHT,WIDTH])
    img = np.array(img)
    img = img/255
    img = img.reshape(-1,HEIGHT,WIDTH,3)
    pr = model.predict(img)[0]
    
    pr = pr.reshape((HEIGHT, WIDTH,NCLASSES)).argmax(axis=-1)
    
    pr = Image.fromarray(np.uint8(pr))
    pr = pr.resize((WIDTH,HEIGHT))
    pr = pr.crop(((WIDTH-nw)//2, (HEIGHT-nh)//2,(WIDTH-nw)//2+nw,(HEIGHT-nh)//2+nh))
    pr = np.array(pr)
    seg_img = np.zeros((nh, nw,3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))

    image = Image.blend(old_img,seg_img,0.3)
    image.save("./img_out/"+jpg)


