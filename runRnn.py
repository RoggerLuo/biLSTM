import argparse
import tensorflow as tf
import pickle
import math
import numpy as np
from sklearn.model_selection import train_test_split
from os.path import join

FLAGS = None
from getY import getY
from getInputs import getInputs
from getLoss import getLoss
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BI LSTM')
    parser.add_argument('--source_data', help='source size', default='./data/data.pkl')
    
    parser.add_argument('--train_batch_size', help='train batch size', default=3)
    parser.add_argument('--dev_batch_size', help='dev batch size', default=3)
    parser.add_argument('--test_batch_size', help='test batch size', default=10)

    parser.add_argument('--num_layer', help='num of layer', default=2, type=int)
    parser.add_argument('--num_units', help='num of units', default=88, type=int)
    parser.add_argument('--embedding_size', help='time steps', default=64, type=int)
    parser.add_argument('--time_step', help='time steps', default=32, type=int)
    parser.add_argument('--category_num', help='category num', default=5, type=int)

    parser.add_argument('--summaries_dir', help='summaries dir', default='summaries/', type=str)
    parser.add_argument('--checkpoint_dir', help='checkpoint dir', default='ckpt/model.ckpt', type=str)

    parser.add_argument('--epoch_num', help='num of epoch', default=10, type=int) #1000
    parser.add_argument('--epochs_per_dev', help='epochs per dev', default=2, type=int)
    parser.add_argument('--epochs_per_save', help='epochs per save', default=2, type=int)

    parser.add_argument('--keep_prob', help='train keep prob dropout', default=0.5, type=float)

    parser.add_argument('--steps_per_print', help='steps per print', default=100, type=int)
    parser.add_argument('--steps_per_summary', help='steps per summary', default=100, type=int)
    parser.add_argument('--learning_rate', help='learning rate', default=0.01, type=float)


    FLAGS, args = parser.parse_known_args()    

    
    x, y_label, train_initializer,dev_initializer = getInputs(FLAGS)
    y,keep_prob = getY(FLAGS,x)
    cross_entropy,accuracy = getLoss(y_label,y)

    train(FLAGS,cross_entropy,accuracy,train_initializer,dev_initializer,keep_prob)



