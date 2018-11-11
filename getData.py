import tensorflow as tf
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(FLAGS):
    """
    Load data from pickle
    :return: Arrays
    """
    with open(FLAGS.source_data, 'rb') as f:
        data_x = pickle.load(f)
        data_y = pickle.load(f)
        word2id = pickle.load(f)
        id2word = pickle.load(f)
        tag2id = pickle.load(f)
        id2tag = pickle.load(f)
        return data_x, data_y, word2id, id2word, tag2id, id2tag

def get_data(data_x, data_y):
    """
    Split data from loaded data
    :param data_x:
    :param data_y:
    :return: Arrays
    """
    
    # print('Data X Length', len(data_x), 'Data Y Length', len(data_y))
    # print('Data X Example', data_x[0])
    # print('Data Y Example', data_y[0])
    '''
    data_x
    [
        [5, 12, 16, 11, 20, 14, 1, 21, 6, 17, 4, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [9, 18, 8, 10, 15, 13, 19, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    等于把中文句子中的字都变成了数字
    data_x是一整个库的句子，很多条
    库是由很多句子组成的
    '''
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=40)
    train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.2, random_state=40)
    
    # print('Train X Shape', train_x.shape, 'Train Y Shape', train_y.shape)
    # print('Dev X Shape', dev_x.shape, 'Dev Y Shape', dev_y.shape)
    # print('Test Y Shape', test_x.shape, 'Test Y Shape', test_y.shape)
    return train_x, train_y, dev_x, dev_y, test_x, test_y

def getData(FLAGS):
    
    # main()

    data_x, data_y, word2id, id2word, tag2id, id2tag = load_data(FLAGS)

    train_x, train_y, dev_x, dev_y, test_x, test_y = get_data(data_x, data_y)

    vocab_size = len(word2id) + 1
    FLAGS.vocab_size = vocab_size

    return train_x, train_y, dev_x, dev_y, test_x, test_y, word2id, id2word, tag2id, id2tag





