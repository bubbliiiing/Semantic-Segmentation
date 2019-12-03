from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
def letterbox_image(image, size, type):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    
    image = image.resize((nw,nh), Image.BICUBIC)
    if(type=="jpg"):
        new_image = Image.new('RGB', size, (0,0,0))
    elif(type=="png"):
        new_image = Image.new('RGB', size, (0,0,0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image,nw,nh

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a
    
def get_random_data(image, label, input_shape, jitter=.2, hue=.2, sat=1.1, val=1.1):
    h, w = input_shape

    # resize image
    rand_jit1 = rand(1-jitter,1+jitter)
    rand_jit2 = rand(1-jitter,1+jitter)
    new_ar = w/h * rand_jit1/rand_jit2
    scale = rand(.6, 1.4)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)
    label = label.resize((nw,nh), Image.BICUBIC)
    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (0,0,0))
    new_label = Image.new('RGB', (w,h), (0,0,0))
    new_image.paste(image, (dx, dy))
    new_label.paste(label, (dx, dy))
    image = new_image
    label = new_label
    # flip image or not
    flip = rand()<.5
    if flip: 
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x)
    return image_data,label

jpg = Image.open("dataset2/jpg/2008_000003.jpg")
png = Image.open("dataset2/png/2008_000003.png")

jpg,_,_ = letterbox_image(jpg,[512,512],"jpg")
png,_,_ = letterbox_image(png,[512,512],"jpg")
jpg.show()

jpg,png = get_random_data(jpg, png, [512,512])
Image.fromarray(np.uint8(jpg * 255)).show()
