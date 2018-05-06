import numpy as np
#import matplotlib.pyplot as plt
#from pylab import *
import os
import sys
import time
import cv2
#from PIL import Image
from keras.preprocessing.image import *
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import keras.backend as K

from models import *
from inference import inference
def decode_segmap(label_mask, plot=False):
        label_colours = get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, 21):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

	
def encode_segmap(mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        #print mask.shape
        label_mask = np.zeros((mask.shape[0],mask.shape[1]), dtype=np.int16)
        #print label_mask.shape
        for ii, label in enumerate(get_pascal_labels()):
            #print label.shape
            #print (np.all(mask == label, axis=-1)).shape
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
            #print label_mask.shape
        label_mask = label_mask.astype(int)
        #print label_mask.shape
        return label_mask
def get_pascal_labels():
	return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                           [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                           [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                           [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                           [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                           [0, 64, 128]])




def calculate_iou(model_name, nb_classes, res_dir, label_dir, image_list):
    conf_m = np.zeros((nb_classes, nb_classes), dtype=float)
    total = 0
    # mean_acc = 0.
    for img_num in image_list:
        img_num = img_num.strip('\n')
        total += 1
       # print('#%d: %s' % (total, img_num))
	img_num=img_num+'.png'
        pred = cv2.imread((os.path.join(res_dir, img_num)))
        label = cv2.imread((os.path.join(label_dir, img_num)))
	pred=pred[:,:,[2,1,0]]
	label=label[:,:,[2,1,0]]
        pred=encode_segmap(pred)
	label=encode_segmap(label)
        flat_pred = np.ravel(pred)
        flat_label = np.ravel(label)
        # acc = 0.
        for p, l in zip(flat_pred, flat_label):
            if l == 255:
                continue
            if l < nb_classes and p < nb_classes:
                conf_m[l, p] += 1
            else:
                print('Invalid entry encountered, skipping! Label: ', l,
                      ' Prediction: ', p, ' Img_num: ', img_num)

        #    if l==p:
        #        acc+=1
        #acc /= flat_pred.shape[0]
        #mean_acc += acc
    #mean_acc /= total
    #print 'mean acc: %f'%mean_acc
    print(conf_m.shape) 
    print(conf_m)
    I = np.diag(conf_m)
    print(I)	
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    print(U)
    IOU = I/U
    meanIOU = np.mean(IOU)
    return conf_m, IOU, meanIOU


def evaluate(model_name, weight_file, image_size, nb_classes, batch_size, val_file_path, data_dir, label_dir,
          label_suffix='.png',
          data_suffix='.jpg'):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(current_dir, 'Models/'+model_name+'/res/')
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    fp = open(val_file_path)
    image_list = fp.readlines()
    fp.close()
    	
    start_time = time.time()
    inference(model_name, weight_file, image_size, image_list, data_dir, label_dir, return_results=False, save_dir=save_dir,
              label_suffix=label_suffix, data_suffix=data_suffix)
    duration = time.time() - start_time
    print('{}s used to make predictions.\n'.format(duration))

    start_time = time.time()
    conf_m, IOU, meanIOU = calculate_iou(model_name, nb_classes, save_dir, label_dir, image_list)
    print('IOU: ')
    print(IOU)
    print('meanIOU: %f' % meanIOU)
    print('pixel acc: %f' % (np.sum(np.diag(conf_m))/np.sum(conf_m)))
    duration = time.time() - start_time
    print('{}s used to calculate IOU.\n'.format(duration))

if __name__ == '__main__':
    # model_name = 'Atrous_DenseNet'
    model_name = 'AtrousFCN_Resnet50_16s'
    model_name='FCN_Vgg16_32s'
    # model_name = 'DenseNet_FCN'
    #model_name='VGGUnet'
   # model_name='VggIFCN'
    weight_file = 'checkpoint_weights.hdf5'
    #weight_file = 'fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    image_size = (512, 512)
    nb_classes = 21
    batch_size = 1
    dataset = 'VOC2012_BERKELEY'
    if dataset == 'VOC2012_BERKELEY':
        # pascal voc + berkeley semantic contours annotations
        train_file_path = '/media/epsilon90/Shasvat/MS Sem2/CMPSCI 690IV/segmentation/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt' #Data/VOClarge/VOC2012/ImageSets/Segmentation
        val_file_path = 'train_sample.txt'
        # data_dir        = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
        data_dir='../data_processed/new/JPEGImages_train_sample/'
	label_dir='../data_processed/SegmentationClass_train_sample/'	
        #data_dir = '/media/epsilon90/Shasvat/MS Sem2/CMPSCI 690IV/segmentation/dataset/VOCdevkit/VOC2012/new/trial'
        # label_dir       = os.path.expanduser('~/.keras/datasets/VOC2012/combined_annotations')
        #label_dir = '/media/epsilon90/Shasvat/MS Sem2/CMPSCI 690IV/segmentation/dataset/VOCdevkit/VOC2012/trial_test'
        label_suffix = '.png'
    if dataset == 'COCO':
        train_file_path = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        # train_file_path = os.path.expanduser('~/.keras/datasets/oneimage/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        val_file_path   = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')
        data_dir        = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
        label_dir       = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/SegmentationClass')
        label_suffix = '.npy'
    evaluate(model_name, weight_file, image_size, nb_classes, batch_size, val_file_path, data_dir, label_dir,
             label_suffix=label_suffix)
