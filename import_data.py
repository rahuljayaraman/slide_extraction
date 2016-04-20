from sequence import Sequence
import numpy as np
from constants import paths
import shutil
import os
import utils

TRAINING_DATA = [
    Sequence("ykatz", "Yehuda Katz",
             "https://www.youtube.com/watch?v=uCaYkUmdtPw",
             "360p", "mp4", [
                 (53, 2595)
             ]),
    Sequence("karpathy1", "Andrej Karpathy",
             "https://www.youtube.com/watch?v=xKt21ucdBY0",
             "360p", "mp4", [
                 (1, 3207)
             ]),
    Sequence("kyle", "Kyle Kastner",
             "https://www.youtube.com/watch?v=TBBtOeY2Q78",
             "360p", "mp4", [
                 (1, 2), (28, 200), (208, 1004)
             ]),
    Sequence("jeremy", "Jeremy Freeman",
             "https://www.youtube.com/watch?v=N17I5FrRTCw",
             "360p", "mp4", [
                 (1, 4434)
             ]),
    Sequence("mostly_negative1", "Stanford biz school",
             "https://www.youtube.com/watch?v=HAnw168huqA",
             "360p", "mp4", [
                 (1, 5), (18, 23), (52, 75), (159, 165), (174, 182),
                 (222, 240), (259, 276), (416, 440), (558, 563), (922, 945),
                 (1571, 1576), (2033, 2040), (2331, 2336), (2376, 2383),
                 (2414, 2421), (2519, 2535), (2578, 2586), (2904, 2916)
             ]),
    Sequence("aaronp", "Aaron Patterson",
             "https://www.youtube.com/watch?v=JMGmaRZtgM8",
             "360p", "mp4", [
                 (58, 82), (86, 125), (142, 170), (179, 186),
                 (197, 210), (221, 229), (248, 262), (279, 280),
                 (302, 303), (340, 381), (388, 408), (414, 418),
                 (430, 435), (443, 448), (456, 461), (469, 474),
                 (482, 487), (525, 526), (547, 547), (585, 587),
                 (596, 597), (656, 676), (680, 682), (690, 695),
                 (703, 703), (716, 721), (732, 734), (744, 747),
                 (768, 773), (781, 785), (794, 799), (807, 812),
                 (822, 824), (846, 851), (859, 864), (872, 877),
                 (885, 885), (911, 916), (924, 929), (937, 942),
                 (950, 955), (963, 964), (968, 968), (976, 977),
                 (980, 981), (989, 994), (1002, 1007), (1015, 1020),
                 (1028, 1033), (1041, 1046), (1054, 1059),
                 (1067, 1072), (1080, 1085), (1093, 1098),
                 (1133, 1137), (1145, 1145), (1171, 1176),
                 (1184, 1189), (1197, 1198), (1212, 1214),
                 (1237, 1241), (1249, 1254), (1265, 1266),
                 (1288, 1293), (1301, 1306), (1315, 1319),
                 (1327, 1332), (1353, 1358), (1381, 1384),
                 (1397, 1397), (1405, 1410), (1418, 1418),
                 (1431, 1436), (1444, 1449), (1470, 1475),
                 (1483, 1488), (1499, 1501), (1509, 1514),
                 (1618, 1618), (1626, 1631), (1639, 1644),
                 (1652, 1657), (1665, 1670), (1678, 1680),
                 (1704, 1709), (1719, 1722), (1730, 1732),
                 (1743, 1748), (1756, 1761), (1769, 1774),
                 (1782, 1787), (1810, 1813), (1821, 1826),
                 (1834, 1839), (1847, 1852), (1886, 1891),
                 (1899, 1904), (1941, 1943), (1951, 1951),
                 (1964, 1969), (1977, 1982), (1990, 1995),
                 (2007, 2008), (2016, 2021)
             ])
]


def serialize(data, labels, batch_size=0):
    class local:
        idx = 0
        batch = 0
        serialized = np.array([], np.uint8)


    def write_batch():
        if not local.serialized.any():
            return

        local.batch += 1
        filename = paths.SERIALIZED_DIR + "data_batch_" + str(local.batch) + ".bin"
        print "Writing file", filename
        local.serialized.tofile(filename)
        local.serialized = np.array([], np.uint8)

    for record in data:
        r = record[:, :, 0].flatten()
        g = record[:, :, 1].flatten()
        b = record[:, :, 2].flatten()
        label = [labels[local.idx]]

        out = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)
        local.serialized = np.concatenate((local.serialized, out), axis=0)

        if batch_size and not local.idx % batch_size:
            write_batch()
        local.idx += 1


    if not batch_size:
        filename = paths.SERIALIZED_DIR + "test_batch.bin"
        print "Writing file", filename
        local.serialized.tofile(filename)
    elif local.serialized.any():
        write_batch()


all_data = {}

for sequence in TRAINING_DATA:
    sequence.download()
    data = sequence.read()
    if not all_data:
        all_data = data
    else:
        for key, value in all_data.iteritems():
            all_data[key] = np.concatenate(
                (all_data[key], data[key]),
                axis=0)

if os.path.exists(paths.SERIALIZED_DIR):
    shutil.rmtree(paths.SERIALIZED_DIR)
utils.print_stats(all_data, ['train', 'test'])
print "Serializing.."
utils.create_dirs_if_not_exists(paths.SERIALIZED_DIR)
serialize(all_data['train_dataset'], all_data['train_labels'], batch_size=2000)
serialize(all_data['test_dataset'], all_data['test_labels'])
