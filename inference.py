import numpy as np
#import matplotlib.pyplot as plt
#from pylab import *
import os
import sys
import cv2
#from PIL import Image
from keras.preprocessing.image import *
from keras.models import load_model
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input

from models import *

def decode_segmap(label_mask, plot=False):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, 21):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r 
        rgb[:, :, 1] = g 
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb
def get_pascal_labels():
	return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                           [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                           [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                           [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                           [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                           [0, 64, 128]])

def inference(model_name, weight_file, image_size, image_list, data_dir, label_dir, return_results=True, save_dir=None,
              label_suffix='.png',
              data_suffix='.jpg'):
    current_dir = os.path.dirname(os.path.realpath(__file__))
   # current_dir='/media/epsilon90/Shasvat/MS Sem2/CMPSCI 690IV/segmentation/Keras-FCN/'
    # mean_value = np.array([104.00699, 116.66877, 122.67892])
    batch_shape = (1, ) + image_size + (3, )
    save_path = os.path.join(current_dir, 'Models/'+model_name)
    model_path = os.path.join(save_path, "model.json")
    checkpoint_path = os.path.join(save_path, weight_file)
    # model_path = os.path.join(current_dir, 'model_weights/fcn_atrous/model_change.hdf5')
    # model = FCN_Resnet50_32s((480,480,3))
    #print(checkpoint_path) 	
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)

    model = globals()[model_name]( input_shape=(512, 512, 3))
    model.load_weights(checkpoint_path, by_name=True)

    model.summary()

    results = []
    total = 0
    for img_num in image_list:
        img_num = img_num.strip('\n')
        total += 1
       # print('#%d: %s' % (total,img_num))
	filename=img_num+data_suffix
        image = cv2.imread(os.path.join(data_dir, filename))
        # image = img_to_array(image)  # , data_format='default')
	#image[:,:,[0,1,2]]=image[:,:,[2,1,0]]
        #print(np.sum(np.sum(image[:,:,0]>0)))
        #print(np.sum(np.sum(image[:,:,1]>0)))
        #print(np.sum(np.sum(image[:,:,2]>0)))

        label = cv2.imread(os.path.join(label_dir, img_num, label_suffix))
        # label_size = label.size

        img_h,img_w = image.shape[0:2]

        # long_side = max(img_h, img_w, image_size[0], image_size[1])
        pad_w = max(image_size[1] - img_w, 0)
        pad_h = max(image_size[0] - img_h, 0)
        #print(pad_w)
        #print(pad_h)
        image = np.lib.pad(image, ((pad_h/2, pad_h - pad_h/2), (pad_w/2, pad_w - pad_w/2), (0, 0)), 'constant', constant_values=0.)
        # image -= mean_value
        '''img = array_to_img(image, 'channels_last', scale=False)
        img.show()
        exit()'''
        # image = cv2.resize(image, image_size)

        image = np.expand_dims(image, axis=0)
        #image = preprocess_input(image)
        #print(image[])
        result = model.predict(image, batch_size=1)
        print(result.shape)
        result = np.argmax(np.squeeze(result), axis=-1).astype(np.uint8)
	#print(result.shape)
	print(np.sum(np.sum(result>0)))
	if(total>1 and total <10):
		print(result)		
	result_img=decode_segmap(result)
	result_img=result_img[:,:,[2,1,0]]
        #print(result_img.shape)
	if total>1 and total <10:
		print(result_img)
	result_img=result_img[pad_h/2:pad_h/2+img_h,pad_w/2:pad_w/2+img_w]
       # result_img = Image.fromarray(result, mode='P')
       # result_img.palette = label.palette
        # result_img = result_img.resize(label_size, resample=Image.BILINEAR)
       # result_img = result_img.crop((pad_w/2, pad_h/2, pad_w/2+img_w, pad_h/2+img_h))
        # result_img.show(title='result')
        if return_results:
            results.append(result_img)
        if save_dir:
            cv2.imwrite((os.path.join(save_dir,img_num+'.png')),result_img)
	    # result_img.save(os.path.join(save_dir, img_num + '.png'))
    return results

if __name__ == '__main__':
    # model_name = 'AtrousFCN_Resnet50_16s'
    # model_name = 'Atrous_DenseNet'
    #model_name = 'DenseNet_FCN'
    model_name = 'FCN_Vgg16_8s'
    weight_file = 'checkpoint_weights.hdf5'
    image_size = (512, 512)
    #data_dir        = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
    #label_dir       = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/SegmentationClass')
    data_dir = '/media/epsilon90/Shasvat/MS Sem2/CMPSCI 690IV/segmentation/dataset/VOCdevkit/VOC2012/new/trial/'
    # label_dir       = os.path.expanduser('~/.keras/datasets/VOC2012/combined_annotations')
    label_dir = '/media/epsilon90/Shasvat/MS Sem2/CMPSCI 690IV/segmentation/dataset/VOCdevkit/VOC2012/trial_test/'

    #image_list = sys.argv[1:]#'2007_000491'
    image_list=['2007_000346']
    results = inference(model_name, weight_file, image_size, image_list, data_dir, label_dir)
    for result in results:
        result.show(title='result', command=None)
