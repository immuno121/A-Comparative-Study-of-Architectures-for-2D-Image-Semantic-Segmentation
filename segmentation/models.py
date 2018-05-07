import numpy as np
#import matplotlib.pyplot as plt
#from pylab import *
import os
import sys
#from keras_contrib.applications import densenet
from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import tensorflow as tf
#from tensorflow.python import keras
#from tensorflow.python.keras.layers import Input, Conv2D, Conv2DTranspose
#from tensorflow.python.keras import backend as K
from utils.get_weights_path import *
from utils.basics import *
from utils.resnet_helpers import *
from utils.BilinearUpSampling import *
from keras.layers import merge, Input
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D, Deconvolution2D, Cropping2D
from keras.models import Model
from keras.engine.topology import Layer
#from keras.utils.layer_utils import layer_from_config
#from keras.utils import np_utils, generic_utils
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input

# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn32s.py
# fc weights into the 1x1 convs  , get_upsampling_weight



from keras.models import *
from keras.layers import *

import os

file_path = os.path.dirname(os.path.abspath(__file__))

#VGG_Weights_path = file_path + "/../data/vgg16_weights_th_dim_ordering_th_kernels.h5"
def VGGUnet( input_shape=None, weight_decay=0., batch_momentum=0
            , batch_shape=None, classes=21):


    
	if batch_shape:
          img_input = Input(batch_shape=batch_shape)
          image_size = batch_shape[1:3]
        else:
          img_input = Input(shape=input_shape)
          image_size = input_shape[0:2]

   
    
        vgg16 = VGG16(include_top=False,
                           weights='imagenet',
                           input_tensor=None,
                           input_shape=(image_size[0], image_size[1],3))

    
        #(samples, channels, rows, cols)
	#ip = Input(shape=(3, self.img_height, self.img_width))
	h = vgg16.layers[1](img_input)
	h = vgg16.layers[2](h)
	h = vgg16.layers[3](h)
	p1=h	
	h = vgg16.layers[4](h)
	h = vgg16.layers[5](h)
	h = vgg16.layers[6](h)
	p2=h
	h = vgg16.layers[7](h)
	h = vgg16.layers[8](h)
	h = vgg16.layers[9](h)
	h = vgg16.layers[10](h)

	# split layer
	p3 = h

	h = vgg16.layers[11](h)
	h = vgg16.layers[12](h)
	h = vgg16.layers[13](h)
	h = vgg16.layers[14](h)

	# split layer
	p4 = h

	h = vgg16.layers[15](h)
	h = vgg16.layers[16](h)
	h = vgg16.layers[17](h)
	h = vgg16.layers[18](h)

	p5 = h
	
	x=h
	x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
	x = Dropout(0.5)(x)
	x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
	x = Dropout(0.5)(x)
 	#p5=x
	x = (Conv2D(classes, (1, 1), kernel_initializer='he_normal',activation='linear',kernel_regularizer=l2(weight_decay)))(x)
	#print(o.shape)
	x = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(x)
	#x = BilinearUpSampling2D(size=(2, 2))(x)
        #print(o.shape)
	#imshow(o)
	o=x
	K.print_tensor(o)
	o2 = p4
	





###########################
	o2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o2)
	#print(o2.shape)
	o, o2 = crop(o, o2, img_input)
	#print(o.shape)
	#print(o2.shape)
	o =(concatenate([ o ,o2],axis=-1)) 


	o = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
	#o = BilinearUpSampling2D(size=(2, 2))(o)
        o2 = p3
	o2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o2)
	o2, o = crop(o2, o, img_input)
	o = (concatenate([ o ,o2],axis=-1 ))	

	#o = Conv2DTranspose(classes, kernel_size=(16, 16), strides=(8, 8), use_bias=False)(o)
	o = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
	#o = BilinearUpSampling2D(size=(2, 2))(o)
        o2 = p2
	o2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o2)
	o2, o = crop(o2, o, img_input)
	o = (concatenate([ o ,o2],axis=-1 ))	
	o = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
        #o = BilinearUpSampling2D(size=(2, 2))(o)

	o2 = p1
	o2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o2)
	o2, o = crop(o2, o, img_input)
	o = (concatenate([ o ,o2],axis=-1 ))	
	o = Conv2DTranspose(classes, kernel_size=(2, 2), strides=(2, 2), use_bias=False)(o)
	#o = BilinearUpSampling2D(size=(2, 2))(o)
        o = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o)


#########3
	# o = (UpSampling2D( (2,2)))(o)
	print("o", o.shape)

	
	o_shape = Model(img_input , o ).output_shape
	print("o shape",o_shape)
	outputHeight = o_shape[1]
	print("output Height = ",outputHeight);
	outputWidth = o_shape[2]
	print("output Width =",outputWidth)
	#outputHeight = 320
	#outputWidth = 320

	o = (Reshape((  classes , outputHeight*outputWidth   )))(o)
	o = (Permute((2, 1)))(o)
	o = (Reshape((  outputHeight,outputWidth,-1  )))(o)
	#o = (Activation('softmax'))(o)
	model = Model( img_input , o )
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight
	#weights_path = os.path.abspath('C:\\Users\\User\\Documents\\UMass Amherst\\Semester 2\\COMPSCI 690IV - Intelligent Visual Computing\\Project - Semantic Segmentation\\Keras-FCN\\Models\\FCN_Vgg16_32s\\vgg16.h5')
	#weights_path='Models/FCN_Vgg16_8s/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'
	#model.load_weights(weights_path, by_name=True)


	return model


#IMAGE_ORDERING = 'channels_first'
############################################################################
def VggIFCN(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    # assert input_height%32 == 0
    # assert input_width%32 == 0

	if batch_shape:
          img_input = Input(batch_shape=batch_shape)
          image_size = batch_shape[1:3]
        else:
          img_input = Input(shape=input_shape)
          image_size = input_shape[0:2]

   
    
        vgg16 = VGG16(include_top=False,
                           weights='imagenet',
                           input_tensor=None,
                           input_shape=(image_size[0], image_size[1],3))

    
        #(samples, channels, rows, cols)
	#ip = Input(shape=(3, self.img_height, self.img_width))
	h = vgg16.layers[1](img_input)
	h = vgg16.layers[2](h)
	h = vgg16.layers[3](h)
	h = vgg16.layers[4](h)
	h = vgg16.layers[5](h)
	h = vgg16.layers[6](h)
	h = vgg16.layers[7](h)
	h = vgg16.layers[8](h)
	h = vgg16.layers[9](h)
	h = vgg16.layers[10](h)

	# split layer
	p3 = h

	h = vgg16.layers[11](h)
	var_3_1=h
	h = vgg16.layers[12](h)
	var_3_2=h
	h = vgg16.layers[13](h)
	var_3_3=h
	h = vgg16.layers[14](h)

	# split layer
	p4 = h

	h = vgg16.layers[15](h)
	var_4_1=h
	h = vgg16.layers[16](h)
	var_4_2=h
	h = vgg16.layers[17](h)
	var_4_3=h
	h = vgg16.layers[18](h)

	p5 = h
	
	x=h
	x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
	x = Dropout(0.5)(x)
	x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
	x = Dropout(0.5)(x)
 	#p5=x
	x = (Conv2D(classes, (1, 1), kernel_initializer='he_normal',activation='linear',kernel_regularizer=l2(weight_decay)))(x)
	#print(o.shape)
	#x = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(x)
        x = BilinearUpSampling2D(size=(2, 2))(x)
	#print(o.shape)
	#imshow(o)
	o=x
	K.print_tensor(o)
	o2 = p4
	o2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o2)
	#print(o2.shape)
	#o, o2 = crop(o, o2, img_input)
	#print(o.shape)
	#print(o2.shape)
	o = Add()([o, o2])
	
	var_4_1 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(var_4_1)
    	var_4_2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(var_4_2)
    	var_4_3 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(var_4_3)
	
	o = Add()([o, var_4_1])
    	o = Add()([o, var_4_2])
    	o = Add()([o, var_4_3])
   	temp_20 = o  # 20X20

    	o = BatchNormalization()(o)

	
	#o = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
	o = BilinearUpSampling2D(size=(2, 2))(o)
	o2 = p3
	o2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o2)
	#o2, o = crop(o2, o, img_input)
	#o = Add()([o2, o])	
	
	o = Add()([o, o2])
 	var_3_1 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(var_3_1)
    	var_3_2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(var_3_2)
    	var_3_3 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(var_3_3)

    # all are of sizes 20X20

    	o = Add()([o, var_3_1])
   	o = Add()([o, var_3_2])
    	o = Add()([o, var_3_3])
    #o = BatchNormalization()(o)
    	temp_40 = o  # 20X20
	




	 #########lets make a context network:#############################################
    	o = p5

    	#o = (Conv2D(4096, (7, 7), activation='relu', padding='same'))(o)
    	#o = Dropout(0.5)(o)
    	# o = (Conv2D(4096, (1, 1), activation='relu', padding='same'))(o)
    	#o = Dropout(0.5)(o)

    	#o = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o)

    	o = (Conv2D(512, (5, 5), kernel_initializer='RandomNormal', padding='same'))(o)
    	o = BatchNormalization()(o)
	
	o = Activation('relu')(o)
    	temp_c1 = o

    	o = (Conv2D(512, (5, 5), kernel_initializer='RandomNormal', padding='same'))(o)
   	o = BatchNormalization()(o)
    	o = Activation('relu')(o)
    	temp_c2 = o

    	o = (Conv2D(512, (5, 5), kernel_initializer='RandomNormal', padding='same'))(o)
    	o = BatchNormalization()(o)
    	o = Activation('relu')(o)
    	temp_c3 = o

    	o = (Conv2D(512, (5, 5), kernel_initializer='RandomNormal', padding='same'))(o)
    	o = BatchNormalization()(o)
    	o = Activation('relu')(o)
    	temp_c4 = o

	o = (Conv2D(512, (5, 5), kernel_initializer='RandomNormal', padding='same'))(o)
        o = BatchNormalization()(o)
        o = Activation('relu')(o)
        temp_c5 = o


    	o = (Conv2D(512, (5, 5), kernel_initializer='RandomNormal', padding='same'))(o)

	o = BatchNormalization()(o)
    	o = Activation('relu')(o)
    	temp_c6 = o

    	o = Add()([o, temp_c1])
    	o = Add()([o, temp_c2])
    	o = Add()([o, temp_c3])
    	o = Add()([o, temp_c4])
    	o = Add()([o, temp_c5])
    	o = Add()([o, temp_c6])
        o=(Conv2D(classes, (1, 1), kernel_initializer='RandomNormal', padding='same'))(o)

	o = BilinearUpSampling2D(size=(2, 2))(o)
    	#o = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
    	#o, temp_20 = crop(o, temp_20, img_input)


	o = Add()([o, temp_20])

    	#temp_20 = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(temp_20)
    	#o = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
	temp_20 = BilinearUpSampling2D(size=(2, 2))(temp_20)
	o = BilinearUpSampling2D(size=(2, 2))(o)
    	#temp_20, temp_40 = crop(temp_20, temp_40, img_input)
    	#o, temp_40 = crop(o, temp_40, img_input)

    	o = Add()([o, temp_40])
    	o = Add()([o, temp_20])

    ####################################################################################

	






	#o = Conv2DTranspose(classes, kernel_size=(16, 16), strides=(8, 8), use_bias=False)(o)
	#o = Conv2DTranspose(classes, kernel_size=(8, 8), strides=(8, 8), use_bias=False)(o)
	o = BilinearUpSampling2D(size=(8, 8))(o)
	o_shape = Model(img_input, o).output_shape

	outputHeight = o_shape[1]
	outputWidth = o_shape[2]

	#o = (Reshape((-1, outputHeight * outputWidth)))(o)
	#o = (Permute((2, 1)))(o)
	#o = (Reshape((outputHeight , outputWidth,-1)))(o)
	#o = (Activation('softmax'))(o)
	model = Model(img_input, o)
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight
	#weights_path = 'Models/FCN_Vgg16_8s/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'
	#model.load_weights(weights_path, by_name=True)
	#return model

	return model





	
	
	# vgg = Model(img_input, x)
	# weights_path = 'Models/FCN_Vgg16_8s/vgg16_weights_th_dim_ordering_th_kernels.h5'

	# vgg.load_weights(weights_path)
	
	
	#################################################################################

# crop o1 wrt o2
def crop(o1, o2, i):
    o_shape2 = Model(i, o2).output_shape
    outputHeight2 = o_shape2[1]
    outputWidth2 = o_shape2[2]

    o_shape1 = Model(i, o1).output_shape
    outputHeight1 = o_shape1[1]
    outputWidth1 = o_shape1[2]
    #print(outputHeight2)
    #print(outputWidth2)
    #print(outputHeight1)
    #print(outputWidth1 )

    cx = abs(outputWidth1 - outputWidth2)
    cy = abs(outputHeight2 - outputHeight1)

    if outputWidth1 > outputWidth2:
        o1 = Cropping2D(cropping=((0, 0), (0, cx)))(o1)
    else:
        o2 = Cropping2D(cropping=((0, 0), (0, cx)))(o2)

    if outputHeight1 > outputHeight2:
        o1 = Cropping2D(cropping=((0, cy), (0, 0)))(o1)
    else:
        o2 = Cropping2D(cropping=((0, cy), (0, 0)))(o2)

    return o1, o2

def FCN_Vgg16_8s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    # assert input_height%32 == 0
    # assert input_width%32 == 0

    # https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
        if batch_shape:
          img_input = Input(batch_shape=batch_shape)
          image_size = batch_shape[1:3]
        else:
          img_input = Input(shape=input_shape)
          image_size = input_shape[0:2]

   
    
        vgg16 = VGG16(include_top=False,
                           weights='imagenet',
                           input_tensor=None,
                           input_shape=(image_size[0], image_size[1],3))

    
        #(samples, channels, rows, cols)
	#ip = Input(shape=(3, self.img_height, self.img_width))
	h = vgg16.layers[1](img_input)
	h = vgg16.layers[2](h)
	h = vgg16.layers[3](h)
	h = vgg16.layers[4](h)
	h = vgg16.layers[5](h)
	h = vgg16.layers[6](h)
	h = vgg16.layers[7](h)
	h = vgg16.layers[8](h)
	h = vgg16.layers[9](h)
	h = vgg16.layers[10](h)

	# split layer
	p3 = h

	h = vgg16.layers[11](h)
	h = vgg16.layers[12](h)
	h = vgg16.layers[13](h)
	h = vgg16.layers[14](h)

	# split layer
	p4 = h

	h = vgg16.layers[15](h)
	h = vgg16.layers[16](h)
	h = vgg16.layers[17](h)
	h = vgg16.layers[18](h)

	p5 = h
	
	x=h
	x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
	x = Dropout(0.5)(x)
	x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
	x = Dropout(0.5)(x)
 	p5=x
	p5 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal',activation='linear',kernel_regularizer=l2(weight_decay)))(p5)
	#print(o.shape)
	#p5 = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(p5)
	#p5=UpSampling2D((2,2))(p5)
	p5 = BilinearUpSampling2D(size=(2, 2))(p5)

	#print(o.shape)
	#imshow(o)
	o=p5
	K.print_tensor(o)
	o2 = p4
	o2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o2)
	#print(o2.shape)
	#o, o2 = crop(o, o2, img_input)
	#print(o.shape)
	#print(o2.shape)
	o = Add()([o, o2])

	o=BilinearUpSampling2D(size=(2, 2))(o)
	#o = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
	o2 = p3
	o2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o2)
	#o2, o = crop(o2, o, img_input)
	o = Add()([o2, o])	

	#o = Conv2DTranspose(classes, kernel_size=(16, 16), strides=(8, 8), use_bias=False)(o)
	#o = Conv2DTranspose(classes, kernel_size=(8, 8), strides=(8, 8), use_bias=False)(o)
	o=BilinearUpSampling2D(size=(8, 8))(o)
	o_shape = Model(img_input, o).output_shape

	outputHeight = o_shape[1]
	outputWidth = o_shape[2]

	#o = (Reshape((-1, outputHeight * outputWidth)))(o)
	#o = (Permute((2, 1)))(o)
	#o = (Reshape((outputHeight , outputWidth,-1)))(o)
	#o = (Activation('softmax'))(o)
	model = Model(img_input, o)
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight
	#weights_path = 'Models/FCN_Vgg16_8s/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'
	#model.load_weights(weights_path, by_name=True)
	#return model

	return model
	











	
	# get scores
	


def top(x, input_shape, classes, activation, weight_decay):

    x = Conv2D(classes, (1, 1), activation='linear',
               padding='same', kernel_regularizer=l2(weight_decay),
               use_bias=False)(x)

    if K.image_data_format() == 'channels_first':
        channel, row, col = input_shape
    else:
        row, col, channel = input_shape

    # TODO(ahundt) this is modified for the sigmoid case! also use loss_shape
    if activation is 'sigmoid':
        x = Reshape((row * col * classes,))(x)

    return x




def FCN_Vgg16_32s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)



    # Block                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    p4=x
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    ###################16s###################################################################
    x = BilinearUpSampling2D(size=(32, 32))(x) 
    #p5=x
    #p5 = BilinearUpSampling2D(size=(2, 2))(p5)

        #print(o.shape)
        #imshow(o)
    #o=p5
    #K.print_tensor(o)
    #o2 = p4
    #o2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o2)
        #print(o2.shape)
        #o, o2 = crop(o, o2, img_input)
        #print(o.shape)
        #print(o2.shape)
    #o = Add()([o, o2])

    #o=BilinearUpSampling2D(size=(16, 16))(o)
    #x=o	
    model = Model(img_input, x)
###############################################16s##########################################################

    #weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    weights_path='Models/FCN_Vgg16_32s/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path, by_name=True)
    return model


def FCN_Vgg16_16s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):

        if batch_shape:
          img_input = Input(batch_shape=batch_shape)
          image_size = batch_shape[1:3]
	  vgg16 = VGG16(include_top=False,
                           weights='imagenet',
                           input_tensor=None,
                           input_shape=(batch_shape[0],image_size[0], image_size[1],3))


        else:
          img_input = Input(shape=input_shape)
          image_size = input_shape[0:2]
          vgg16 = VGG16(include_top=False,
                           weights='imagenet',
                           input_tensor=None,
                           input_shape=(image_size[0], image_size[1],3))

    
        #(samples, channels, rows, cols)
	#ip = Input(shape=(3, self.img_height, self.img_width))
	h = vgg16.layers[1](img_input)
	h = vgg16.layers[2](h)
	h = vgg16.layers[3](h)
	h = vgg16.layers[4](h)
	h = vgg16.layers[5](h)
	h = vgg16.layers[6](h)
	h = vgg16.layers[7](h)
	h = vgg16.layers[8](h)
	h = vgg16.layers[9](h)
	h = vgg16.layers[10](h)

	# split layer
	p3 = h

	h = vgg16.layers[11](h)
	h = vgg16.layers[12](h)
	h = vgg16.layers[13](h)
	h = vgg16.layers[14](h)

	# split layer
	p4 = h

	h = vgg16.layers[15](h)
	h = vgg16.layers[16](h)
	h = vgg16.layers[17](h)
	h = vgg16.layers[18](h)

	p5 = h
	
	x=h
	x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
	x = Dropout(0.5)(x)
	x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
	x = Dropout(0.5)(x)
 	p5=x
	p5 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal',activation='linear',kernel_regularizer=l2(weight_decay)))(p5)
	#print(o.shape)
	#p5 = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(p5)
	#p5=UpSampling2D((2,2))(p5)
	p5 = BilinearUpSampling2D(size=(2, 2))(p5)

	#print(o.shape)
	#imshow(o)
	o=p5
	K.print_tensor(o)
	o2 = p4
	o2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o2)
	#print(o2.shape)
	#o, o2 = crop(o, o2, img_input)
	#print(o.shape)
	#print(o2.shape)
	o = Add()([o, o2])

	o=BilinearUpSampling2D(size=(16, 16))(o)
	#o = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
	#o2 = p3
	#o2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o2)
	#o2, o = crop(o2, o, img_input)
	#o = Add()([o2, o])	

	#o = Conv2DTranspose(classes, kernel_size=(16, 16), strides=(8, 8), use_bias=False)(o)
	#o = Conv2DTranspose(classes, kernel_size=(8, 8), strides=(8, 8), use_bias=False)(o)
	#o=BilinearUpSampling2D(size=(8, 8))(o)
	o_shape = Model(img_input, o).output_shape

	outputHeight = o_shape[1]
	outputWidth = o_shape[2]

	#o = (Reshape((-1, outputHeight * outputWidth)))(o)
	#o = (Permute((2, 1)))(o)
	#o = (Reshape((outputHeight , outputWidth,-1)))(o)
	#o = (Activation('softmax'))(o)
	model = Model(img_input, o)
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight
	#weights_path = 'Models/FCN_Vgg16_8s/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'
	#model.load_weights(weights_path, by_name=True)
	#return model

	return model
	








def FCN_Resnet50_32s(input_shape = None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]

    bn_axis = 3

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', strides=(1, 1))(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b')(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c')(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d')(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f')(x)

    x = conv_block(3, [512, 512, 2048], stage=5, block='a')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='b')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='c')(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(size=(32, 32))(x)

    model = Model(img_input, x)
    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
    model.load_weights(weights_path, by_name=True)
    return model


def AtrousFCN_Resnet50_16s(input_shape = None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]

    bn_axis = 3

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', weight_decay=weight_decay, strides=(1, 1), batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = atrous_conv_block(3, [512, 512, 2048], stage=5, block='a', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='b', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='c', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    #classifying layer
    #x = Conv2D(classes, (3, 3), dilation_rate=(2, 2), kernel_initializer='normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = Model(img_input, x)
    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
    model.load_weights(weights_path, by_name=True)
    return model


def Atrous_DenseNet(input_shape=None, weight_decay=1E-4,
                    batch_momentum=0.9, batch_shape=None, classes=21,
                    include_top=False, activation='sigmoid'):
    # TODO(ahundt) pass the parameters but use defaults for now
    if include_top is True:
        # TODO(ahundt) Softmax is pre-applied, so need different train, inference, evaluate.
        # TODO(ahundt) for multi-label try per class sigmoid top as follows:
        # x = Reshape((row * col * classes))(x)
        # x = Activation('sigmoid')(x)
        # x = Reshape((row, col, classes))(x)
        return densenet.DenseNet(depth=None, nb_dense_block=3, growth_rate=32,
                                 nb_filter=-1, nb_layers_per_block=[6, 12, 24, 16],
                                 bottleneck=True, reduction=0.5, dropout_rate=0.2,
                                 weight_decay=1E-4,
                                 include_top=True, top='segmentation',
                                 weights=None, input_tensor=None,
                                 input_shape=input_shape,
                                 classes=classes, transition_dilation_rate=2,
                                 transition_kernel_size=(1, 1),
                                 transition_pooling=None)

    # if batch_shape:
    #     img_input = Input(batch_shape=batch_shape)
    #     image_size = batch_shape[1:3]
    # else:
    #     img_input = Input(shape=input_shape)
    #     image_size = input_shape[0:2]

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=16,
                                      data_format=K.image_data_format(),
                                      include_top=False)
    img_input = Input(shape=input_shape)

    x = densenet.__create_dense_net(classes, img_input,
                                    depth=None, nb_dense_block=3, growth_rate=32,
                                    nb_filter=-1, nb_layers_per_block=[6, 12, 24, 16],
                                    bottleneck=True, reduction=0.5, dropout_rate=0.2,
                                    weight_decay=1E-4, top='segmentation',
                                    input_shape=input_shape,
                                    transition_dilation_rate=2,
                                    transition_kernel_size=(1, 1),
                                    transition_pooling=None,
                                    include_top=include_top)

    x = top(x, input_shape, classes, activation, weight_decay)

    model = Model(img_input, x, name='Atrous_DenseNet')
    # TODO(ahundt) add weight loading
    return model
def DenseNet_FCN(input_shape=None, weight_decay=1E-4,
                 batch_momentum=0.9, batch_shape=None, classes=21,
                 include_top=False, activation='sigmoid'):
    if include_top is True:
        # TODO(ahundt) Softmax is pre-applied, so need different train, inference, evaluate.
        # TODO(ahundt) for multi-label try per class sigmoid top as follows:
        # x = Reshape((row * col * classes))(x)
        # x = Activation('sigmoid')(x)
        # x = Reshape((row, col, classes))(x)
        return densenet.DenseNetFCN(input_shape=input_shape,
                                    weights=None, classes=classes,
                                    nb_layers_per_block=[4, 5, 7, 10, 12, 15],
                                    growth_rate=16,
                                    dropout_rate=0.2)

    # if batch_shape:
    #     img_input = Input(batch_shape=batch_shape)
    #     image_size = batch_shape[1:3]
    # else:
    #     img_input = Input(shape=input_shape)
    #     image_size = input_shape[0:2]

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=16,
                                      data_format=K.image_data_format(),
                                      include_top=False)
    img_input = Input(shape=input_shape)

    x = densenet.__create_fcn_dense_net(classes, img_input,
                                        input_shape=input_shape,
                                        nb_layers_per_block=[4, 5, 7, 10, 12, 15],
                                        growth_rate=16,
                                        dropout_rate=0.2,
                                        include_top=include_top)

    x = top(x, input_shape, classes, activation, weight_decay)
    # TODO(ahundt) add weight loading
    model = Model(img_input, x, name='DenseNet_FCN')
    return model
