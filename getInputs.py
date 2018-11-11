import argparse
import tensorflow as tf
import pickle
import math
import numpy as np
from sklearn.model_selection import train_test_split
from os.path import join
from getBatch import getBatch


def getInputs(FLAGS):    

    train_dataset,dev_dataset,test_dataset = getBatch(FLAGS)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_initializer = iterator.make_initializer(train_dataset)
    dev_initializer = iterator.make_initializer(dev_dataset)
    test_initializer = iterator.make_initializer(test_dataset)
    # Input Layer
    with tf.variable_scope('inputs'):
        x, y_label = iterator.get_next()

    return x, y_label, train_initializer,dev_initializer




