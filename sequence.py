from pytube import YouTube
import os.path
import subprocess as sp
import itertools
import imageio
import numpy as np
import shutil
import utils
from constants import defaults
from constants import paths

IMAGE_SIZE = defaults.IMAGE_SIZE


class Sequence:

    def __init__(self, id, label, url, res, fmt, slices):
        self.id = id
        self.label = label
        self.url = url
        self.res = res
        self.fmt = fmt

        self.dir_name = paths.DATA_DIR + '/' + self.id
        self.file_path = self.dir_name + "/" + self.id + "." + self.fmt

        yt = YouTube(url)
        yt.set_filename(id)
        self.downloader = yt.get(fmt, res)

        # flatten
        self.positives = list(itertools.chain(
            *[range(x[0], x[1]) for x in slices]))

    def download(self):
        utils.create_dirs_if_not_exists(self.dir_name)

        if os.path.isfile(self.file_path):
            print "file: %s already exists, skipping" % self.file_path
        else:
            print "downloading %s to %s" % (self.url, self.file_path)
            self.downloader.download(self.dir_name)

    def read(self):
        print "reading %s" % self.file_path
        TEMP = paths.TEMP_DIR + '/images/'
        try:
            shutil.rmtree(TEMP)
        except OSError as e:
            pass
        utils.create_dirs_if_not_exists(TEMP)

        # TODO: use piplines instead of savings files to disk
        command = ['ffmpeg', '-n', '-i', self.file_path,
                   '-vf', 'fps=1, scale=' +
                   str(IMAGE_SIZE) + ':' + str(IMAGE_SIZE),
                   TEMP + '%d.png']
        sp.call(command)

        image_files = os.listdir(TEMP)
        max_examples = len(image_files)

        dataset = np.ndarray(shape=(max_examples, IMAGE_SIZE,
                                    IMAGE_SIZE, 3), dtype=np.float32)
        labels = np.ndarray(shape=(max_examples), dtype=np.int32)

        idx = 0
        for file in image_files:
            seq = int(file.split(".")[0])
            src = TEMP + file
            try:
                image = imageio.imread(src)
                if image.shape != (IMAGE_SIZE,
                                   IMAGE_SIZE, 3):
                    raise Exception('Image incorrectly resized',
                                    src, image.shape)
                dataset[idx] = image
                if seq in self.positives:
                    labels[idx] = 1
                else:
                    labels[idx] = 0
                idx += 1
            except IOError as e:
                print "Could not read", src, e

        # Remove unused space
        dataset = dataset[0:idx, :, :]
        labels = labels[0:idx]

        sample_range = self.__random_sample(dataset)
        print "creating test set from", sample_range[0], "to", sample_range[1]
        test_dataset, test_labels = self.__extract_dataset(dataset,
                                                           labels,
                                                           sample_range)
        dataset, labels = self.__delete_range(dataset, labels, sample_range)
        shutil.rmtree(TEMP)

        return {
            'train_dataset': dataset,
            'train_labels': labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels
        }

    def __random_sample(self, dataset):
        rand_idx = np.random.randint(1, 4)
        samples, step = np.linspace(0, len(dataset), num=5,
                                    dtype=np.int32, retstep=True)
        return samples[rand_idx], samples[rand_idx] + step

    def __extract_dataset(self, dataset, labels, sample_range):
        test_dataset = dataset[sample_range[0]:sample_range[1], :, :]
        test_labels = labels[sample_range[0]:sample_range[1]]
        return test_dataset, test_labels

    def __delete_range(self, dataset, labels, sample_range):
        dataset = np.delete(dataset, np.arange(sample_range[0],
                                               sample_range[1]), axis=0)
        labels = np.delete(labels, np.arange(sample_range[0],
                                             sample_range[1]), axis=0)
        return dataset, labels
