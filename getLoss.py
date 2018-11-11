import tensorflow as tf
import pickle
import math
import numpy as np
from os.path import join

def getLoss(y_label,y):

    y_predict = tf.cast(tf.argmax(y, axis=1), tf.int32)
    print('Output Y.shape:', y_predict.shape)    

    tf.summary.histogram('y_predict', y_predict)

    y_label_reshape = tf.cast(tf.reshape(y_label, [-1]), tf.int32)
    print('Y Label Reshape:', y_label_reshape)

    correct_prediction = tf.equal(y_predict, y_label_reshape)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    print('Prediction:', correct_prediction, 'Accuracy', accuracy)
    
    # Loss
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label_reshape,
                                                                                  logits=tf.cast(y, tf.float32)))
    tf.summary.scalar('loss', cross_entropy)
    print('cross_entropy:')
    print(cross_entropy)
    return cross_entropy,accuracy
