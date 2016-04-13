from dataset import DataSet
from constants import Constants
import os
import cPickle as pickle
import numpy as np
import tensorflow as tf

class DataSets(object):
    pass

def read_all_data():
    listing = os.listdir(Constants.DATA_DIR)
    all_data = []

    for item in listing:
        if item.endswith('pickle'):
            filename = Constants.DATA_DIR + '/' + item
            with open(filename) as f:
                data = pickle.load(f)
                if not all_data:
                    all_data = data
                else:
                    for key, value in all_data.iteritems():
                        all_data[key] = np.concatenate((all_data[key], data[key]), axis=0)


    return all_data

def unroll4d(data):
    shape = data.shape
    return data.reshape(shape[0], shape[1] * shape[2] * shape[3])


def read_ann_data_sets(dtype=tf.float32):
    data_sets = DataSets()

    data = read_all_data()

    train_images = unroll4d(data['train_dataset'])
    train_labels = data['train_labels']

    test_images = unroll4d(data['test_dataset'])
    test_labels = data['test_labels']

    validation_images = unroll4d(data['validation_dataset'])
    validation_labels = data['validation_labels']

    data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
    data_sets.validation = DataSet(validation_images, validation_labels, dtype=dtype)
    data_sets.test = DataSet(test_images, test_labels, dtype=dtype)

    return data_sets

def print_stats(dataset, labeldata):
    for l in labeldata:
        data = dataset[l + '_dataset']
        label = dataset[l + '_labels']
        print 'Data Shape: ', l, data.shape
        print 'Label Shape: ', l, label.shape
        print 'Mean: ', l, np.mean(data)
        print 'Std dev: ', l, np.std(data)
