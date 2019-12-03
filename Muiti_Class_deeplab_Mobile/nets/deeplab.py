from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Activation
from keras.layers import Softmax,Reshape
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D
from keras.layers import GlobalAveragePooling2D
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.activations import relu
from keras.applications.imagenet_utils import preprocess_input
from nets.mobilenetV2 import mobilenetV2

def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    # 计算padding的数量，hw是否需要收缩
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'
    
    # 如果需要激活函数
    if not depth_activation:
        x = Activation('relu')(x)

    # 分离卷积，首先3x3分离卷积，再1x1卷积
    # 3x3采用膨胀卷积
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    # 1x1卷积，进行压缩
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x

def Deeplabv3(input_shape=(416, 416, 3), classes=21, alpha=1.):

    img_input = Input(shape=input_shape)

    # (52, 52, 320)
    x,skip1 = mobilenetV2(img_input,alpha)
    size_before = tf.keras.backend.int_shape(x)

    # 全部求平均后，再利用expand_dims扩充维度，1x1
    # shape = 320
    b4 = GlobalAveragePooling2D()(x)
    
    # 1x1x320
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    
    # 压缩filter
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)

    # 直接利用resize_images扩充hw
    # b4 = 52,52,256
    b4 = Lambda(lambda x: tf.image.resize_images(x, size_before[1:3]))(b4)
    # 调整通道
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # rate值与OS相关，SepConv_BN为先3x3膨胀卷积，再1x1卷积，进行压缩
    # 其膨胀率就是rate值
    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=6, depth_activation=True, epsilon=1e-5)
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=12, depth_activation=True, epsilon=1e-5)
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=18, depth_activation=True, epsilon=1e-5)

    x = Concatenate()([b4, b0, b1, b2, b3])

    # 利用conv2d压缩
    # 52,52,256
    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)


    # skip1.shape[1:3] 为 104,104
    # skip1 104, 104, 256
    x = Lambda(lambda xx: tf.image.resize_images(x, skip1.shape[1:3]))(x)
                                                    
    # 104, 104, 48
    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                        use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(
        name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)

    # 104,104,304
    x = Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, 256, 'decoder_conv0',
                    depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1',
                    depth_activation=True, epsilon=1e-5)

    # 416,416,2
    size_before3 = tf.keras.backend.int_shape(img_input)

    # 104,104,2
    x = Conv2D(classes, (1, 1), padding='same')(x)
    x = Lambda(lambda xx:tf.image.resize_images(xx,size_before3[1:3]))(x)

    x = Reshape((-1,classes))(x)
    x = Softmax()(x)

    inputs = img_input
    model = Model(inputs, x, name='deeplabv3plus')

    return model

