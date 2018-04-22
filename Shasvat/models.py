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

    '''
    input_height = 320
    input_width = 320
    img_input = Input(shape=(3,input_height,input_width))
    '''

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    temp_x=x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x
    print(f1.shape)
	# Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x
    print(f2.shape)
	# Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x
    print(f3.shape)
	# Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x
    print(f4.shape)
   	# Block 5
   # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
   # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
   # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
   # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool' )(x)
   # f5 = x
   # print(f5.shape)
   # x = Flatten(name='flatten')(x)
   # x = Dense(4096, activation='relu', name='fc1')(x)
   # x = Dense(4096, activation='relu', name='fc2')(x)
   # x = Dense( 1000 , activation='softmax', name='predictions')(x)
   # print(x.shape)
    #vgg  = Model(  img_input , x  )
    #vgg.load_weights(VGG_Weights_path)
    #weights_path = os.path.abspath('C:\\Users\\User\\Documents\\UMass Amherst\\Semester 2\\COMPSCI 690IV - Intelligent Visual Computing\\Project - Semantic Segmentation\\Keras-FCN\\Models\\FCN_Vgg16_32s\\vgg16.h5')
    #vgg.load_weights(weights_path, by_name=True)
    #levels = [f1 , f2 , f3 , f4 , f5 ]

    o = f4
    print("o=f4",o.shape)
    o = ( ZeroPadding2D( (1,1)))(o)
    print("0-padding",o.shape)
    o = ( Conv2D(512, (3, 3), padding='valid'))(o)
    print("conv2d",o.shape)
    o = BatchNormalization()(o)
    print("batch-norm",o.shape)

   # o = (UpSampling2D( (2,2)))(o)
    o=Conv2DTranspose(256,kernel_size=(2,2),strides=(2,2),padding='same')(o)	
    print("upsample",o.shape)
    print("f3",f3.shape)
    o = ( concatenate([ o ,f3],axis=-1 )  )
    o = ( ZeroPadding2D( (1,1)))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid'))(o)
    o = ( BatchNormalization())(o)
    o=Conv2DTranspose(128,kernel_size=(2,2),strides=(2,2),padding='same')(o)	

   # o = (UpSampling2D( (2,2)))(o)
    o = ( concatenate([o,f2],axis=-1 ) )
    o = ( ZeroPadding2D((1,1)))(o)
    o = ( Conv2D( 128 , (3, 3), padding='valid') )(o)
    o = ( BatchNormalization())(o)
    o=Conv2DTranspose(64,kernel_size=(2,2),strides=(2,2),padding='same')(o)	

   # o = (UpSampling2D( (2,2)))(o)
    o = ( concatenate([o,f1],axis=-1 ) )
    o = ( ZeroPadding2D((1,1)))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'))(o)
    o = ( BatchNormalization())(o)
    o=Conv2DTranspose(64,kernel_size=(2,2),strides=(2,2),padding='same')(o)	
     	
   # o = (UpSampling2D( (2,2)))(o)
    print("o", o.shape)

    n_classes=classes
    o =  Conv2D( n_classes , (3, 3) , padding='same')( o )
    o_shape = Model(img_input , o ).output_shape
    print("o shape",o_shape)
    outputHeight = o_shape[1]
    print("output Height = ",outputHeight);
    outputWidth = o_shape[2]
    print("output Width =",outputWidth)
    #outputHeight = 320
    #outputWidth = 320

    o = (Reshape((  n_classes , outputHeight*outputWidth   )))(o)
    o = (Permute((2, 1)))(o)
    o = (Reshape((  outputHeight,outputWidth,-1  )))(o)
    o = (Activation('softmax'))(o)
    model = Model( img_input , o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    #weights_path = os.path.abspath('C:\\Users\\User\\Documents\\UMass Amherst\\Semester 2\\COMPSCI 690IV - Intelligent Visual Computing\\Project - Semantic Segmentation\\Keras-FCN\\Models\\FCN_Vgg16_32s\\vgg16.h5')
    weights_path='Models/FCN_Vgg16_8s/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path, by_name=True)


    return model


#IMAGE_ORDERING = 'channels_first'
############################################################################
def VggIFCN(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    # assert input_height%32 == 0
    # assert input_width%32 == 0

    # https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
        img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    var_3_1 = x
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    var_3_2 = x
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    var_3_3 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    var_4_1 = x
    print("var4.1", var_4_1.shape)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    var_4_2 = x
    print("var4.2", var_4_2.shape)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    var_4_3 = x
    print("var4.3", var_4_3.shape)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1000, activation='softmax', name='predictions')(x)

    # vgg = Model(img_input, x)
    # weights_path = 'Models/FCN_Vgg16_8s/vgg16_weights_th_dim_ordering_th_kernels.h5'
    # vgg.load_weights(weights_path)

    o = f5

    o = (Conv2D(4096, (7, 7), activation='relu', padding='same'))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(4096, (1, 1), activation='relu', padding='same'))(o)
    o = Dropout(0.5)(o)

    o = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o)
    # o:size=10X10
    o = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
    # o:size=22X22
    o2 = f4
    o2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o2)
    # o2:size=20X20
    o, o2 = crop(o, o2, img_input)
    # both are 20X20

    o = Add()([o, o2])

    var_4_1 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(var_4_1)
    var_4_2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(var_4_2)
    var_4_3 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(var_4_3)

    # all are of sizes 20X20

    o = Add()([o, var_4_1])
    o = Add()([o, var_4_2])
    o = Add()([o, var_4_3])
    temp_20 = o  # 20X20
    o = BatchNormalization()(o)
    o = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
    # o:size=44X44

    o2 = f3

    o2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o2)
    # o2:size=40X40
    o, o2 = crop(o, o2, img_input)

    # Both are 40X40
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

    # print(o.shape)

    # print(o.shape)
    # imshow(o)
    K.print_tensor(o)
    #o=BatchNormalization()(o)
    #########lets make a context network:#############################################
    o = f5

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

    o = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
    o, temp_20 = crop(o, temp_20, img_input)
    o = Add()([o, temp_20])

    temp_20 = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(temp_20)
    o = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)

    temp_20, temp_40 = crop(temp_20, temp_40, img_input)
    o, temp_40 = crop(o, temp_40, img_input)

    o = Add()([o, temp_40])
    o = Add()([o, temp_20])

    ####################################################################################

    # o = Conv2DTranspose(classes, kernel_size=(16, 16), strides=(8, 8), use_bias=False)(o)
    o = Conv2DTranspose(classes, kernel_size=(8, 8), strides=(8, 8), use_bias=False)(o)

    o_shape = Model(img_input, o).output_shape

    outputHeight = o_shape[1]
    outputWidth = o_shape[2]

    o = (Reshape((-1, outputHeight * outputWidth)))(o)
    o = (Permute((2, 1)))(o)
    o = (Reshape((outputHeight, outputWidth, -1)))(o)
    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    weights_path = 'Models/FCN_Vgg16_8s/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path, by_name=True)
    # return model

    return model
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

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
        img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x

   # x = Flatten(name='flatten')(x)
   # x = Dense(4096, activation='relu', name='fc1')(x)
   # x = Dense(4096, activation='relu', name='fc2')(x)
   # x = Dense(1000, activation='softmax', name='predictions')(x)

    #vgg = Model(img_input, x)
    #weights_path = 'Models/FCN_Vgg16_8s/vgg16_weights_th_dim_ordering_th_kernels.h5'
    #vgg.load_weights(weights_path)

    o = f5

    o = (Conv2D(4096, (7, 7), activation='relu', padding='same',name='fc1',kernel_regularizer=l2(weight_decay)))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(4096, (1, 1), activation='relu', padding='same',name='fc2',kernel_regularizer=l2(weight_decay)))(o)
    o = Dropout(0.5)(o)

    o = (Conv2D(classes, (1, 1), kernel_initializer='he_normal',activation='linear',kernel_regularizer=l2(weight_decay)))(o)
    #print(o.shape)
    o = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
    #print(o.shape)
    #imshow(o)
    K.print_tensor(o)
    o2 = f4
    o2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o2)
    #print(o2.shape)
    o, o2 = crop(o, o2, img_input)
    #print(o.shape)
    #print(o2.shape)
    o = Add()([o, o2])

    o = Conv2DTranspose(classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
    o2 = f3
    o2 = (Conv2D(classes, (1, 1), kernel_initializer='he_normal'))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add()([o2, o])

    #o = Conv2DTranspose(classes, kernel_size=(16, 16), strides=(8, 8), use_bias=False)(o)
    o = Conv2DTranspose(classes, kernel_size=(8, 8), strides=(8, 8), use_bias=False)(o)

    o_shape = Model(img_input, o).output_shape

    outputHeight = o_shape[1]
    outputWidth = o_shape[2]

    o = (Reshape((-1, outputHeight * outputWidth)))(o)
    o = (Permute((2, 1)))(o)
    o = (Reshape((outputHeight , outputWidth,-1)))(o)
    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    weights_path = 'Models/FCN_Vgg16_8s/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path, by_name=True)
    #return model

    return model



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

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)



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

    x = BilinearUpSampling2D(size=(32, 32))(x)

    model = Model(img_input, x)


    #weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    weights_path='Models/FCN_Vgg16_32s/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path, by_name=True)
    return model


def AtrousFCN_Vgg16_16s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
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

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', dilation_rate=(2, 2),
                      name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = Model(img_input, x)

    weights_path = 'Models/FCN_Vgg16_32s/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path, by_name=True)
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
