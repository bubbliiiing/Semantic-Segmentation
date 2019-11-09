from keras.models import *
from keras.layers import *
from nets.mobilenet import get_mobilenet_encoder


IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1


def _unet( n_classes , encoder , l1_skip_conn=True,  input_height=416, input_width=608  ):

	img_input , levels = encoder( input_height=input_height ,  input_width=input_width )
	[f1 , f2 , f3 , f4 , f5 ] = levels 

	o = f4
	# 26,26,512
	o = ( ZeroPadding2D( (1,1) , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	# 52,52,512
	o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	# 52,52,768
	o = ( concatenate([ o ,f3],axis=MERGE_AXIS )  )
	o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
	# 52,52,256
	o = ( Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	# 104,104,256
	o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	# 104,104,384
	o = ( concatenate([o,f2],axis=MERGE_AXIS ) )
	o = ( ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING ))(o)
	# 104,104,128
	o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format=IMAGE_ORDERING ) )(o)
	o = ( BatchNormalization())(o)
	# 208,208,128
	o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	
	if l1_skip_conn:
		o = ( concatenate([o,f1],axis=MERGE_AXIS ) )

	o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING ))(o)
	o = ( BatchNormalization())(o)

	o =  Conv2D( n_classes , (3, 3) , padding='same', data_format=IMAGE_ORDERING )( o )
	
	# 将结果进行reshape
	o = Reshape((int(input_height/2)*int(input_width/2), -1))(o)
	o = Softmax()(o)
	model = Model(img_input,o)

	return model



def mobilenet_unet( n_classes ,  input_height=224, input_width=224 , encoder_level=3):

	model =  _unet( n_classes , get_mobilenet_encoder ,  input_height=input_height, input_width=input_width  )
	model.model_name = "mobilenet_unet"
	return model

