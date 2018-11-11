import argparse
import tensorflow as tf
import pickle
import math
import numpy as np
from sklearn.model_selection import train_test_split
from os.path import join



def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)


def bias(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)


def lstm_cell(num_units, keep_prob=0.5):
    cell = tf.nn.rnn_cell.LSTMCell(num_units, reuse=tf.AUTO_REUSE)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)


def getY(FLAGS,x):        
    print('FLAGS.vocab_size:')
    print(FLAGS.vocab_size)
    print('FLAGS.embedding_size:')
    print(FLAGS.embedding_size)

    # Embedding Layer
    with tf.variable_scope('embedding'):
        embedding = tf.Variable(tf.random_normal([FLAGS.vocab_size, FLAGS.embedding_size]), dtype=tf.float32)
        print('embedding', embedding)

    inputs = tf.nn.embedding_lookup(embedding, x)
    print('inputs.shape:')
    print(inputs.shape)
    
    keep_prob = tf.placeholder(tf.float32, [])
    # FLAGS.keep_prob
    # keep_prob = FLAGS.keep_prob
    # RNN Layer
    cell_fw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]
    cell_bw = [lstm_cell(FLAGS.num_units, keep_prob) for _ in range(FLAGS.num_layer)]

    print('FLAGS.time_step:(这个time_step需要和batch中每个句子的字符长度一致)')
    print(FLAGS.time_step)

    
    inputs = tf.unstack(inputs, FLAGS.time_step, axis=1)
    print('inputs length after unstack:')
    print(len(inputs))
    print('inputs[0].shape length after unstack:')
    print(inputs[0].shape)


    output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, inputs=inputs, dtype=tf.float32)
    


    print('output length (== time_step):')
    print(len(output))

    print('FLAGS.num_units:')
    print(FLAGS.num_units)

    print('output[0].shape ( == double num_units):')
    print(output[0].shape)


    output = tf.stack(output, axis=1)
    print('output.shape after stack:')
    print(output.shape)

    output = tf.reshape(output, [-1, FLAGS.num_units * 2])
    print('output after Reshape')
    print(output.shape)

    with tf.variable_scope('outputs'):
        w = weight([FLAGS.num_units * 2, FLAGS.category_num])
        b = bias([FLAGS.category_num])
        y = tf.matmul(output, w) + b

    print('FLAGS.category_num')
    print(FLAGS.category_num)

    print('y.shape')
    print(y.shape)
    return y,keep_prob
