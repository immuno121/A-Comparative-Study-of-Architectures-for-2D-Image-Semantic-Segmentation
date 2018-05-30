import keras.backend as K
import tensorflow as tf
from tensorflow.contrib.metrics import streaming_mean_iou
#from matplotlib import pyplot as plt
#import train
import numpy as np
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
    #r=label_mask
    #g=r
    #b=r
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
def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    #print nb_classes
    #plt.imshow(y_pred)
    #plt.show()
    #
    #y_temp = (K.argmax(y_pred, axis=-1))
    #sess = tf.Session()
    #with sess.as_default():
        #tensor = tf.constant(np_array)
        #print(tensor)
    #y_temp = K.eval(y_temp)
        #print(numpy_array_2)


    #print K.shape(y_temp)
    #y_temp=y_temp.eval()
    #decode_segmap(np.array(y_temp,dtype='uint8'), plot=True)
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)
    #print type(y_true)
    #print type(y_pred)

    #assert isinstance(y_pred, object)
    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))

# This IOU implementation is wrong!!!
'''def mean_iou_ignoring_last_label(y_true, y_pred):
    batch_size = K.int_shape(y_pred)[0]
    y_true_list = tf.unpack(y_true, num=batch_size, axis=0)
    y_pred_list = tf.unpack(y_pred, num=batch_size, axis=0)
    mean_iou = 0.
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        nb_classes = K.int_shape(y_pred)[-1]
        y_pred = K.reshape(y_pred, (-1, nb_classes))
        y_pred = K.argmax(y_pred, axis=-1)
        y_pred = K.one_hot(y_pred, nb_classes)
        y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), nb_classes + 1)
        unpacked = tf.unpack(y_true, axis=-1)
        legal_labels = tf.expand_dims(tf.to_float(
            ~tf.cast(unpacked[-1], tf.bool)), -1)
        y_true = tf.pack(unpacked[:-1], axis=-1)
        y_true = K.argmax(y_true, axis=-1)
        y_true = K.one_hot(y_true, nb_classes)
        y_pred = tf.cast(y_pred, tf.bool)
        y_true = tf.cast(y_true, tf.bool)

        intersection = tf.to_float(y_pred & y_true) * legal_labels
        union = tf.to_float(y_pred | y_true) * legal_labels
        intersection = K.sum(intersection, axis=0)
        union = K.sum(union, axis=0)
        total_union = K.sum(tf.to_float(tf.cast(union, tf.bool)))
        iou = K.sum(intersection / (union + K.epsilon())) / total_union
        mean_iou = mean_iou + iou
    mean_iou = mean_iou / batch_size
    return mean_iou'''
