import tensorflow as tf
from keras.layers import *
from keras.models import *

from nets.mobilenet import get_mobilenet_encoder

def resize_image(inp, s):
	return Lambda(lambda x: tf.image.resize_images(x, (K.int_shape(x)[1]*s[0], K.int_shape(x)[2]*s[1])))( inp )

def pool_block( feats , pool_factor ):
	h = K.int_shape( feats )[1]
	w = K.int_shape( feats )[2]

    #-----------------------------------------------------#
	# 	strides = [18, 18],[9, 9],[6, 6],[3, 3]
	# 	进行不同程度的平均
    #-----------------------------------------------------#
	pool_size = strides = [int(np.round(float(h)/pool_factor)),int(np.round(float(w)/pool_factor))]
	x = AveragePooling2D(pool_size, strides=strides, padding='same')(feats)
	
    #-----------------------------------------------------#
    #   利用1x1卷积进行通道数的调整
    #-----------------------------------------------------#
	x = Conv2D(512, (1, 1), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

    #-----------------------------------------------------#
    #   利用resize扩大特征层面积
    #-----------------------------------------------------#
	x = resize_image(x, strides) 
	return x

def _pspnet(n_classes, encoder,  input_height=576, input_width=576):
	assert input_height%192 == 0
	assert input_width%192 == 0

	img_input , levels = encoder( input_height=input_height,input_width=input_width)
	[f1 , f2 , f3 , f4 , f5] = levels 

	o = f5
    #--------------------------------------------------------------#
    #	PSP模块，分区域进行池化
    #   分别分割成1x1的区域，2x2的区域，3x3的区域，6x6的区域
    #--------------------------------------------------------------#
	pool_factors = [1,2,3,6]
	pool_outs = [o]
	for p in pool_factors:
		pooled = pool_block(o, p)
		pool_outs.append(pooled)
    #-----------------------------------------------------------------------------------------#
    #   利用获取到的特征层进行堆叠
    #   18, 18, 1024 + 18, 18, 512 + 18, 18, 512 + 18, 18, 512 + 18, 18, 512 = 18, 18, 3072
    #-----------------------------------------------------------------------------------------#
	o = Concatenate()(pool_outs )

	# 18, 18, 3072 -> 36, 36, 512
	o = Conv2D(512, (1, 1), padding='valid')(o)
	o = BatchNormalization()(o)
	o = resize_image(o, (2,2))
	# 36, 36, 512 + 36, 36, 512 -> 36, 36, 1024
	o = Concatenate()([o,f4])

	# 36, 36, 1024 -> 36, 36, 512
	o = Conv2D(512, (1, 1), padding='valid')(o)
	o = BatchNormalization()(o)
	o = Activation('relu' )(o)

    #--------------------------------------------------------------#
    #	PSP模块，分区域进行池化
    #   分别分割成1x1的区域，2x2的区域，3x3的区域，6x6的区域
    #--------------------------------------------------------------#
	pool_factors = [1,2,3,6]
	pool_outs = [o]
	for p in pool_factors:
		pooled = pool_block(o, p)
		pool_outs.append(pooled)
    #-----------------------------------------------------------------------------------------#
    #   利用获取到的特征层进行堆叠
    #   36, 36, 512 + 36, 36, 512 + 36, 36, 512 + 36, 36, 512 + 36, 36, 512 = 36, 36, 2560
    #-----------------------------------------------------------------------------------------#
	o = Concatenate()(pool_outs)
	
	# 36, 36, 2560 -> 72, 72, 512
	o = Conv2D(512, (1, 1), padding='valid')(o)
	o = BatchNormalization()(o)
	o = resize_image(o, (2,2))
	# 72, 72, 512 + 72, 72, 256 -> 72, 72, 768
	o = Concatenate()([o,f3])

	# 72, 72, 768 -> 72, 72, 512
	o = Conv2D(512, (1, 1), padding='valid')(o)
	o = BatchNormalization()(o)
	o = Activation('relu')(o)

    #--------------------------------------------------------------#
    #	PSP模块，分区域进行池化
    #   分别分割成1x1的区域，2x2的区域，3x3的区域，6x6的区域
    #--------------------------------------------------------------#
	pool_factors = [1,2,3,6]
	pool_outs = [o]
	for p in pool_factors:
		pooled = pool_block(o, p)
		pool_outs.append(pooled)
    #-----------------------------------------------------------------------------------------#
    #   利用获取到的特征层进行堆叠
    #   72, 72, 512 + 72, 72, 512 + 72, 72, 512 + 72, 72, 512 + 72, 72, 512 = 72, 72, 2560
    #-----------------------------------------------------------------------------------------#
	o = Concatenate()(pool_outs)

	# 72, 72, 2560 -> 72, 72, 512
	o = Conv2D(512, (1,1), use_bias=False )(o)
	o = BatchNormalization()(o)
	o = Activation('relu' )(o)

	# 72, 72, 512 -> 144,144,nclasses
	o = Conv2D(n_classes, (3,3), padding='same')(o)
	o = resize_image(o, (2,2))
	o = Reshape((-1, n_classes))(o)
	o = Softmax()(o)
	model = Model(img_input,o)
	return model

def mobilenet_pspnet(n_classes, input_height=224, input_width=224 ):
	model = _pspnet(n_classes, get_mobilenet_encoder, input_height=input_height, input_width=input_width)
	model.model_name = "mobilenet_pspnet"
	return model
