from keras.models import *
from keras.layers import *
from nets.convnet import get_convnet_encoder

IMAGE_ORDERING = 'channels_last'
def segnet_decoder(  f , n_classes , n_up=3 ):

	assert n_up >= 2

	o = f
	# 26,26,512
	o = ( ZeroPadding2D( (1,1) , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	# 进行一次UpSampling2D，此时hw变为原来的1/8
	# 52,52,256
	o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
	o = ( Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	# 进行一次UpSampling2D，此时hw变为原来的1/4
	# 104,104,128
	for _ in range(n_up-2):
		o = ( UpSampling2D((2,2)  , data_format=IMAGE_ORDERING ) )(o)
		o = ( ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING ))(o)
		o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format=IMAGE_ORDERING ))(o)
		o = ( BatchNormalization())(o)

	# 进行一次UpSampling2D，此时hw变为原来的1/2
	# 208,208,64
	o = ( UpSampling2D((2,2)  , data_format=IMAGE_ORDERING ))(o)
	o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING ))(o)
	o = ( BatchNormalization())(o)

	# 此时输出为h_input/2,w_input/2,nclasses
	# 208,208,2
	o =  Conv2D( n_classes , (3, 3) , padding='same', data_format=IMAGE_ORDERING )( o )
	
	return o 

def _segnet( n_classes , encoder  ,  input_height=416, input_width=416 , encoder_level=3):
	# encoder通过主干网络
	img_input , levels = encoder( input_height=input_height ,  input_width=input_width )

	# 获取hw压缩四次后的结果
	feat = levels[encoder_level]

	# 将特征传入segnet网络
	o = segnet_decoder(feat, n_classes, n_up=3 )

	# 将结果进行reshape
	o = Reshape((int(input_height/2)*int(input_width/2), -1))(o)
	o = Softmax()(o)
	model = Model(img_input,o)

	return model

def convnet_segnet( n_classes ,  input_height=224, input_width=224 , encoder_level=3):

	model = _segnet( n_classes , get_convnet_encoder ,  input_height=input_height, input_width=input_width , encoder_level=encoder_level)
	model.model_name = "convnet_segnet"
	return model

