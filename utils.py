import os
import numpy as np

def create_dirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_stats(dataset, labeldata):
    for l in labeldata:
        data = dataset[l + '_dataset']
        label = dataset[l + '_labels']
        print 'Data Shape: ', l, data.shape
        print 'Label Shape: ', l, label.shape
        print 'Mean: ', l, np.mean(data)
        print 'Std dev: ', l, np.std(data)
