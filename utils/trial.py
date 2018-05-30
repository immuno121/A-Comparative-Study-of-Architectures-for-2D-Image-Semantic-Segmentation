import keras.backend as K
import tensorflow as tf
from tensorflow.contrib.metrics import streaming_mean_iou
import numpy as np

def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    print nb_classes
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    print K.int_shape(y_pred)
    #0print shape(y_true)
    print K.int_shape(K.flatten(y_true))
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    print K.int_shape(y_true)
    unpacked = tf.unstack(y_true, axis=-1)
    print K.int_shape(y_true)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    print K.int_shape(y_true)
    y_true = tf.stack(unpacked[:-1], axis=-1)
    print type(y_true)
    print type(y_pred)
    #assert isinstance(y_pred, object)
    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))
input = K.placeholder(shape=(100, 100, 21))
y_true=np.zeros((100,100,3))
#y_pred=np.zeros((100,100,1))
sparse_accuracy_ignoring_last_label(y_true, input)