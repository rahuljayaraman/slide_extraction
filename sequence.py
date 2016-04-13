from pytube import YouTube
import os.path
import subprocess as sp
import itertools
import imageio
import numpy as np
import cPickle as pickle
import shutil
from constants import Constants


class Sequence:

    def __init__(self, id, label, url, res, fmt, slices):
        self.id = id
        self.label = label
        self.url = url
        self.res = res
        self.fmt = fmt

        self.dir_name = Constants.DATA_DIR + '/' + self.id
        self.file_path = self.dir_name + "/" + self.id + "." + self.fmt

        yt = YouTube(url)
        yt.set_filename(id)
        self.downloader = yt.get(fmt, res)
        self.serialized_path = Constants.DATA_DIR + "/" + self.id + ".pickle"

        #flatten
        self.positives = list(itertools.chain(*[range(x[0], x[1]) for x in slices]))

    def read(self):
        try:
            with open(self.serialized_path) as f:
                return pickle.load(f)
        except IOError:
            self.download()
            self.serialize()
            return self.read()

    def download(self):
        self.__create_dirs_if_not_exists(self.dir_name)

        if os.path.isfile(self.file_path):
            print "file: %s already exists, skipping" % self.file_path
        else:
            print "downloading %s to %s" % (self.url, self.file_path)
            self.downloader.download(self.dir_name)

    def serialize(self):
        print "serializing %s" % self.file_path
        TEMP = "./tmp/images/"
        self.__create_dirs_if_not_exists(TEMP)


        #TODO: use piplines instead of savings files to disk
        command = ['ffmpeg', '-n', '-i', self.file_path, 
                   '-vf', 'fps=1, scale=' + str(Constants.IMAGE_SIZE) + ':' + str(Constants.IMAGE_SIZE),
                   TEMP + '%d.png']
        sp.call(command)

        image_files = os.listdir(TEMP)
        max_examples = len(image_files)

        dataset = np.ndarray(shape=(max_examples, Constants.IMAGE_SIZE, Constants.IMAGE_SIZE, 3), dtype=np.float32)
        labels = np.ndarray(shape=(max_examples), dtype=np.int32)

        idx = 0
        for file in image_files:
            seq = int(file.split(".")[0])
            src = TEMP + file
            try:
                image = self.__normalize(imageio.imread(src))
                if image.shape != (Constants.IMAGE_SIZE, Constants.IMAGE_SIZE, 3):
                    raise Exception('Image incorrectly resized', src, image.shape)
                dataset[idx] = self.__normalize(imageio.imread(src))
                if seq in self.positives:
                    labels[idx] = 1
                else:
                    labels[idx] = 0
                idx = idx + 1
            except IOError as e:
                print "Could not read", src

        #Remove unused space
        dataset = dataset[0:idx, :, :]
        labels = labels[0:idx]

        sample_range = self.__random_sample(dataset)
        print "creating test set from", sample_range[0], "to", sample_range[1]
        test_dataset, test_labels = self.__extract_dataset(dataset, labels, sample_range)
        dataset, labels = self.__delete_range(dataset, labels, sample_range)

        sample_range = self.__random_sample(dataset)
        print "creating validation set from", sample_range[0], "to", sample_range[1]
        validation_dataset, validation_labels = self.__extract_dataset(dataset, labels, sample_range)
        dataset, labels = self.__delete_range(dataset, labels, sample_range)

        try:
            f = open(self.serialized_path, 'wb')
            save = {
                'train_dataset': dataset,
                'train_labels': labels,
                'test_dataset': test_dataset,
                'test_labels': test_labels,
                'validation_dataset': validation_dataset,
                'validation_labels': validation_labels
            }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print "Unable to save data to", self.serialized_path, ':', e
            raise

        shutil.rmtree(TEMP)

    def __random_sample(self, dataset):
        rand_idx = np.random.randint(1, 4)
        samples, step = np.linspace(0, len(dataset), num=5, dtype=np.int32, retstep=True)
        return samples[rand_idx], samples[rand_idx] + step
    
    def __extract_dataset(self, dataset, labels, sample_range):
        test_dataset = dataset[sample_range[0]:sample_range[1], :, :]
        test_labels = labels[sample_range[0]:sample_range[1]]
        return test_dataset, test_labels

    def __delete_range(self, dataset, labels, sample_range):
        dataset = np.delete(dataset, np.arange(sample_range[0], sample_range[1]), axis=0)
        labels = np.delete(labels, np.arange(sample_range[0], sample_range[1]), axis=0)
        return dataset, labels

    def __normalize(self, image):
        PIXEL_DEPTH = 255
        return (image - PIXEL_DEPTH/2)/PIXEL_DEPTH
    
    def __create_dirs_if_not_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
