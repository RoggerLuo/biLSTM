import argparse
import tensorflow as tf
import pickle
import math
import numpy as np

from getData import getData


def getBatch(FLAGS):
    train_x, train_y, dev_x, dev_y, test_x, test_y, word2id, id2word, tag2id, id2tag = getData(FLAGS)
    
    train_steps = math.ceil(train_x.shape[0] / FLAGS.train_batch_size)
    dev_steps = math.ceil(dev_x.shape[0] / FLAGS.dev_batch_size)
    test_steps = math.ceil(test_x.shape[0] / FLAGS.test_batch_size)

    FLAGS.train_steps = train_steps
    FLAGS.dev_steps = dev_steps
    FLAGS.test_steps = test_steps

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.batch(FLAGS.train_batch_size)
    
    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_x, dev_y))
    dev_dataset = dev_dataset.batch(FLAGS.dev_batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_dataset = test_dataset.batch(FLAGS.test_batch_size)


    return train_dataset,dev_dataset,test_dataset


