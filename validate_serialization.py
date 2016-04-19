import os
import numpy as np
from PIL import Image
from constants import paths
from constants import defaults

IMAGE_SIZE = defaults.IMAGE_SIZE

data = open(paths.SERIALIZED_DIR + "data_batch_1.bin","rb") 

for i in range(1000,1010):
    label = data.read(1)
    # TODO: Convert label to ascii & print
    image = data.read(IMAGE_SIZE * IMAGE_SIZE * 3)
    c_image = np.frombuffer(image,dtype=np.uint8)
    c_image = c_image.reshape([3, IMAGE_SIZE * IMAGE_SIZE]).T.reshape([IMAGE_SIZE, IMAGE_SIZE,3])
    c_image = Image.fromarray(c_image,mode="RGB")
    c_image.save(open("test_" + str(i) + ".png","w"))
