import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.activations import relu
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, DepthwiseConv2D, Dropout,
                          GlobalAveragePooling2D, Input, Lambda, Reshape,
                          Softmax, ZeroPadding2D)
from keras.models import Model
from keras.utils.data_utils import get_file

from nets.mobilenetV2 import mobilenetV2


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'
    
    if not depth_activation:
        x = Activation('relu')(x)

    # 首先使用3x3的深度可分离卷积
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    # 利用1x1卷积进行通道数调整
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x

def Deeplabv3(input_shape=(416, 416, 3), classes=21, alpha=1.):
    img_input = Input(shape=input_shape)

    # x         52, 52, 320
    # skip1     104, 104, 24
    x, skip1 = mobilenetV2(img_input, alpha)
    size_before = tf.keras.backend.int_shape(x)

    #---------------------------------------------------------------#
    #   全部求平均后，再利用expand_dims扩充维度
    #   52,52,320 -> 1,1,320 -> 1,1,320
    #---------------------------------------------------------------#
    b4 = GlobalAveragePooling2D()(x)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    # 1,1,256 -> 52,52,256
    b4 = Lambda(lambda x: tf.image.resize_images(x, size_before[1:3]))(b4)

    #---------------------------------------------------------------#
    #   调整通道
    #---------------------------------------------------------------#
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # 52, 52, 256 + 52, 52, 256 -> 52, 52, 512
    x = Concatenate()([b4, b0])

    # 利用1x1卷积调整通道数
    # 52, 52, 1280 -> 52,52,256
    x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # 52,52,256 -> 104,104,2 -> 416,416,2
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Conv2D(classes, (1, 1), padding='same')(x)
    x = Lambda(lambda xx:tf.image.resize_images(xx, size_before3[1:3]))(x)

    x = Reshape((-1,classes))(x)
    x = Softmax()(x)

    model = Model(img_input, x, name='deeplabv3plus')
    return model

