import numpy as np
#import matplotlib.pyplot as plt
#from pylab import *
import cv2
#from PIL import Image
import os
import sys
import pickle
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import *
from keras.objectives import *
from keras.metrics import binary_accuracy
from keras.models import load_model
import keras.backend as K
#import keras.utils.visualize_util as vis_util

from models import *
from utils.loss_function import *
from utils.metrics import *
from utils.SegDataGenerator import *
import time


def train(batch_size, epochs, lr_base, lr_power, weight_decay, classes,
          model_name, train_file_path, val_file_path,
          data_dir, label_dir, target_size=None, batchnorm_momentum=0.9,
          resume_training=False, class_weight=None, dataset='VOC2012',
          loss_fn=softmax_sparse_crossentropy_ignoring_last_label,
          metrics=[sparse_accuracy_ignoring_last_label],
          loss_shape=None,
          label_suffix='.png',
          data_suffix='.jpg',
          ignore_label=255,
          label_cval=255):
    if target_size:
        input_shape = target_size + (3,)
    else:
        input_shape = (None, None, 3)
    batch_shape = (batch_size,) + input_shape

    ###########################################################
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + model_name)
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)

    # ###############learning rate scheduler####################
    def lr_scheduler(epoch, mode='power_decay'):
        '''if lr_dict.has_key(epoch):
            lr = lr_dict[epoch]
            print 'lr: %f' % lr'''

        if mode is 'power_decay':
            # original lr scheduler
            lr = lr_base * ((1 - float(epoch)/epochs) ** lr_power)
        if mode is 'exp_decay':
            # exponential decay
            lr = (float(lr_base) ** float(lr_power)) ** float(epoch+1)
        # adam default lr
        if mode is 'adam':
            lr = 0.001

        if mode is 'progressive_drops':
            # drops as progression proceeds, good for sgd
            if epoch > 0.9 * epochs:
                lr = 0.0001
            elif epoch > 0.75 * epochs:
                lr = 0.001
            elif epoch > 0.5 * epochs:
                lr = 0.01
            else:
                lr = 0.1

       # print('lr: %f' % lr)
        return lr
    scheduler = LearningRateScheduler(lr_scheduler)

    # ###################### make model ########################
    checkpoint_path = os.path.join(save_path, 'checkpoint_weights.hdf5')

    model = globals()[model_name](weight_decay=weight_decay,
                                  input_shape=input_shape,
                                  batch_momentum=batchnorm_momentum,
                                  classes=classes,
                                  )

    # ###################### optimizer ########################
    optimizer = SGD(lr=lr_base, momentum=0.9)
    #optimizer = Nadam(lr=lr_base, beta_1 = 0.825, beta_2 = 0.99685)

    model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=metrics)
    if resume_training:
        model.load_weights(checkpoint_path, by_name=True)
    model_path = os.path.join(save_path, "model.json")
    # save model structure
    f = open(model_path, 'w')
    model_json = model.to_json()
    f.write(model_json)
    f.close
    img_path = os.path.join(save_path, "model.png")
    # #vis_util.plot(model, to_file=img_path, show_shapes=True)
    model.summary()

    # lr_reducer      = ReduceLROnPlateau(monitor=softmax_sparse_crossentropy_ignoring_last_label, factor=np.sqrt(0.1),
    #                                     cooldown=0, patience=15, min_lr=0.5e-6)
    # early_stopper   = EarlyStopping(monitor=sparse_accuracy_ignoring_last_label, min_delta=0.0001, patience=70)
    # callbacks = [early_stopper, lr_reducer]
    callbacks = [scheduler]

    # ####################### tfboard ###########################
    if K.backend() == 'tensorflow':
        tensorboard = TensorBoard(log_dir=os.path.join(save_path, 'logs'), histogram_freq=0, write_graph=True)
        callbacks.append(tensorboard)
    # ################### checkpoint saver#######################
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'checkpoint_weights.hdf5'), save_weights_only=True)#.{epoch:d}
    #print 'hi'
    callbacks.append(checkpoint)
    # set data generator and train

    train_datagen = SegDataGenerator(zoom_range=[0.5, 2.0],
                                     zoom_maintain_shape=True,
                                     crop_mode='random',
                                     crop_size=target_size,
                                     # pad_size=(505, 505),
                                     rotation_range=0.,
                                     shear_range=0,
                                     horizontal_flip=True,
                                     channel_shift_range=20.,
                                     fill_mode='constant',
                                     label_cval=label_cval)

    val_datagen = SegDataGenerator()
    train_datagen = SegDataGenerator()
    def load_test_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):
            #img = cv2.imread(os.path.join(folder, filename))
            img=Image.open(os.path.join(folder, filename))

            img = img.resize((320, 320))
            img = np.array(img, dtype='uint32')
            img = img[...,np.newaxis]
            #print img.shape
            #print img_to_array(img=img)
            if img is not None:
                images.append(img)
        return images
    def load_train_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))

            img = cv2.resize(img, (512, 512))
            #print img_to_array(img=img)
            if img is not None:
                images.append(img)
        return images
    def get_file_len(file_path):
        fp = open(file_path)
        lines = fp.readlines()
        fp.close()
        return len(lines)

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
        """Load the mapping that associates pascal classes with label colors
        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                           [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                           [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                           [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                           [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                           [0, 64, 128]])

    def encode_segmap(mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
	#print(mask.shape)
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
       # print( label_mask.shape)
	#print('label')
        return label_mask


    X = load_train_images_from_folder('../data_processed/new/JPEGImages_train/')
    y = load_train_images_from_folder('../data_processed/SegmentationClass/')
  # from Keras documentation: Total number of steps (batches of samples) to yield from generator before declaring one epoch finished
    # and starting the next epoch. It should typically be equal to the number of unique samples of your dataset divided by the batch size.
   # steps_per_epoch = int(np.ceil(get_file_len(train_file_path) / float(batch_size)))
   # print(len(X))
   # print(len(y))
    #print('data loaded')	
    X = np.array(X)
    y = np.array(y)
   # print y.shape
    y_en=np.zeros((y.shape[0],y.shape[1],y.shape[2]))	
    #print X.shape
    #print type(X)
    #print type(y)
   # print((X[0]).shape)
    #print(y[0].shape)
    y_temp=np.zeros((512,512))
    #print(y[0,:,:,:].shape)
    y[0,:,:,[0,1,2]]=y[0,:,:,[2,1,0]]	
    y_temp[:,:]=encode_segmap(y[0])
    #print(y_temp.shape)	
    #print(y_temp)
    #print((y_temp>0).sum())		
    y_orig=decode_segmap(y_temp)	
    #print(y_orig.shape)	
    y_orig[:,:,[0,1,2]]=y_orig[:,:,[2,1,0]]
    cv2.imwrite('test.png',y_orig)
    y[0,:,:,[2,1,0]]=y[0,:,:,[0,1,2]]	
    for i in range(y.shape[0]):
	 #print(y[i].shape)		
   	 y[i,:,:,[0,1,2]]=y[i,:,:,[2,1,0]]	
 	 y_en[i,:,:]=encode_segmap(y[i])
         #y_temp[:,:]=y_en[i,:,:] 
         #print(np.sum(np.sum(y_temp>0))) 
         #y_orig[:,:,[0,1,2]]=y_orig[:,:,[2,1,0]]
         #cv2.imwrite('test.png',y_orig)

   # print y_en.shape
    #print type()
    #print y_en.shape

    #y_enp=np.reshape(y_en,(320,320,1))
    y_en = y_en[..., np.newaxis]

    model.fit(X, y_en, epochs=50, batch_size=16,callbacks=callbacks)


    '''
    history = model.fit_generator(
        generator=train_datagen.flow_from_directory(
            file_path=train_file_path,
            data_dir=data_dir, data_suffix=data_suffix,
            label_dir=label_dir, label_suffix=label_suffix,
            classes=classes,
            target_size=target_size, color_mode='rgb',
            batch_size=batch_size, shuffle=True,
            loss_shape=loss_shape,
            ignore_label=ignore_label,
            # save_to_dir='Images/'
        ),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        workers=4,
        # validation_data=val_datagen.flow_from_directory(
        #     file_path=val_file_path, data_dir=data_dir, data_suffix='.jpg',
        #     label_dir=label_dir, label_suffix='.png',classes=classes,
        #     target_size=target_size, color_mode='rgb',
        #     batch_size=batch_size, shuffle=False
        # ),
        # nb_val_samples = 64
        class_weight=class_weight
       )
    '''
    model.save_weights(save_path+'/model.hdf5')

if __name__ == '__main__':
    #model_name = 'AtrousFCN_Resnet50_16s'
    #model_name = 'Atrous_DenseNet'
    #model_name = 'DenseNet_FCN'
   # model_name='FCN_Vgg16_8s'
    model_name='FCN_Vgg16_32s'
    #model_name='VggIFCN'
    #model_name='VGGUnet'
    batch_size = 16
    batchnorm_momentum = 0.95
    epochs = 250
    lr_base = 0.0001 * (float(batch_size) / 16)
    lr_power = 0.9
    resume_training =True
    if model_name is 'AtrousFCN_Resnet50_16s':
        weight_decay = 0.0001/2
    else:
        weight_decay = 1e-4
    target_size = (320,320)
    target_size = (512, 512)
    dataset = 'VOC2012_BERKELEY'
    if dataset == 'VOC2012_BERKELEY':
        # pascal voc + berkeley semantic contours annotations
        #train_file_path = os.path.expanduser('~/.keras/datasets/VOC2012/combined_imageset_train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        train_file_path='/media/epsilon90/Shasvat/MS Sem2/CMPSCI 690IV/segmentation/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
        # train_file_path = os.path.expanduser('~/.keras/datasets/oneimage/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        #val_file_path   = os.path.expanduser('~/.keras/datasets/VOC2012/combined_imageset_val.txt')
        val_file_path='/media/epsilon90/Shasvat/MS Sem2/CMPSCI 690IV/segmentation/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
        #data_dir        = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
        data_dir='/media/epsilon90/Shasvat/MS Sem2/CMPSCI 690IV/segmentation/dataset/VOCdevkit/VOC2012/JPEGImages/'
        #label_dir       = os.path.expanduser('~/.keras/datasets/VOC2012/combined_annotations')
        label_dir='/media/epsilon90/Shasvat/MS Sem2/CMPSCI 690IV/segmentation/dataset/VOCdevkit/VOC2012/SegmentationClass/'
        data_suffix='.jpg'
        label_suffix='.png'
        classes = 21
    if dataset == 'COCO':
        # ###################### loss function & metric ########################
        train_file_path = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        # train_file_path = os.path.expanduser('~/.keras/datasets/oneimage/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        val_file_path   = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')
        data_dir        = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
        label_dir       = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/SegmentationClass')
        loss_fn = binary_crossentropy_with_logits
        metrics = [binary_accuracy]
        loss_shape = (target_size[0] * target_size[1] * classes,)
        label_suffix = '.npy'
        data_suffix='.jpg'
        ignore_label = None
        label_cval = 0


    # ###################### loss function & metric ########################
    if dataset == 'VOC2012' or dataset == 'VOC2012_BERKELEY':
        loss_fn = softmax_sparse_crossentropy_ignoring_last_label
        metrics = [sparse_accuracy_ignoring_last_label]
        loss_shape = None
        ignore_label = 255
        label_cval = 255

    # Class weight is not yet supported for 3+ dimensional targets
    # class_weight = {i: 1 for i in range(classes)}
    # # The background class is much more common than all
    # # others, so give it less weight!
    # class_weight[0] = 0.1
    class_weight = None

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)
    train(batch_size, epochs, lr_base, lr_power, weight_decay, classes, model_name, train_file_path, val_file_path,
          data_dir, label_dir, target_size=target_size, batchnorm_momentum=batchnorm_momentum, resume_training=resume_training,
          class_weight=class_weight, loss_fn=loss_fn, metrics=metrics, loss_shape=loss_shape, data_suffix=data_suffix,
          label_suffix=label_suffix, ignore_label=ignore_label, label_cval=label_cval)
