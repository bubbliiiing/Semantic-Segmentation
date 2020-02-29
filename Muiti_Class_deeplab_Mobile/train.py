from nets.deeplab import Deeplabv3
from keras.utils.data_utils import get_file
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
import time
import keras
from keras import backend as K
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

ALPHA = 1.0
WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"

                    
NCLASSES = 21
HEIGHT = 512
WIDTH = 512

def letterbox_image(image, size, type):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    
    image = image.resize((nw,nh), Image.NEAREST)
    if(type=="jpg"):
        new_image = Image.new('RGB', size, (0,0,0))
    elif(type=="png"):
        new_image = Image.new('RGB', size, (0,0,0))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image,nw,nh

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(image, label, input_shape, jitter=.1, hue=.1, sat=1.1, val=1.1):

    h, w = input_shape

    # resize image
    rand_jit1 = rand(1-jitter,1+jitter)
    rand_jit2 = rand(1-jitter,1+jitter)
    new_ar = w/h * rand_jit1/rand_jit2
    scale = rand(.7, 1.3)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.NEAREST)
    label = label.resize((nw,nh), Image.NEAREST)
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


def generate_arrays_from_file(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            jpg = Image.open(r".\dataset2\jpg" + '/' + name)
            jpg,_,_ = letterbox_image(jpg,(WIDTH,HEIGHT),"jpg")
            
            name = (lines[i].split(';')[1]).replace("\n", "")
            # 从文件中读取图像
            png = Image.open(r".\dataset2\png" + '/' + name)
            png,_,_ = letterbox_image(png,(HEIGHT,WIDTH),"png")

            jpg, png = get_random_data(jpg,png,[WIDTH,HEIGHT])

            png = png.resize([int(HEIGHT),int(WIDTH)])
            png = np.array(png)
            seg_labels = np.zeros((int(HEIGHT),int(WIDTH),NCLASSES))
            for c in range(NCLASSES):
                seg_labels[: , : , c ] = (png[:,:,0] == c ).astype(int)
            seg_labels = np.reshape(seg_labels, (-1,NCLASSES))

            X_train.append(jpg)
            Y_train.append(seg_labels)

            # 读完一个周期后重新开始
            i = (i+1) % n
        yield (np.array(X_train),np.array(Y_train))

def loss(y_true, y_pred):
    crossloss = K.binary_crossentropy(y_true,y_pred)
    loss = K.sum(crossloss)/HEIGHT/WIDTH
    return loss

if __name__ == "__main__":
    log_dir = "logs/"
    # 获取model
    model = Deeplabv3(classes=NCLASSES,input_shape=(HEIGHT,WIDTH,3))
    # model.summary()

    weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_MOBILE,
                                    cache_subdir='models')
    # model.load_weights(weights_path,by_name=True,skip_mismatch=True)
    model.load_weights(weights_path,by_name=True,skip_mismatch=True)
    # 打开数据集的txt
    with open(r".\dataset2\train_data.txt","r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    # 保存的方式，3世代保存一次
    checkpoint_period = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', 
                                    save_weights_only=True, 
                                    save_best_only=True, 
                                    period=1
                                )
    # 学习率下降的方式，val_loss 2次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='val_loss', 
                            factor=0.5, 
                            patience=2, 
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            min_delta=0, 
                            patience=6, 
                            verbose=1
                        )
    # 交叉熵
    model.compile(loss = loss,
            optimizer = Adam(lr=1e-3),
            metrics = ['accuracy'])
    batch_size = 2
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=30,
            initial_epoch=0,
            callbacks=[checkpoint_period, reduce_lr, early_stopping])

    model.save_weights(log_dir+'middle1.h5')

    # 交叉熵
    model.compile(loss = loss,
            optimizer = Adam(lr=1e-4),
            metrics = ['accuracy'])
    batch_size = 2
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=60,
            initial_epoch=30,
            callbacks=[checkpoint_period, reduce_lr, early_stopping])

    model.save_weights(log_dir+'last1.h5')
