import os
import numpy as np
from os.path import isfile, join

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

def remove_files_only(dir_path):
    onlyfiles = [f for f in os.listdir(dir_path) if isfile(join(dir_path, f))]
    for file in onlyfiles:
        os.remove(join(dir_path, file))
