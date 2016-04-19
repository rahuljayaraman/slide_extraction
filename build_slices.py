import sys
import os.path

if not len(sys.argv) > 1:
    print "Please specify folder"
    sys.exit()

folder_name = sys.argv[1]

positive_folder = folder_name + '/1'
negative_folder = folder_name + '/0'

if not (os.path.exists(positive_folder) and os.path.exists(negative_folder)):
    print "Incorrect folder specified, %s" % folder_name
    sys.exit()

files = []
for file in os.listdir(positive_folder):
    if file.endswith(".png"):
        files.append(int(file.split(".")[0]))

sets = []
start = prev_idx = 0

for seq in sorted(files):
    if prev_idx == 0:
        start = prev_idx = seq
        continue

    if seq == prev_idx + 1:
        prev_idx += 1
    else:
        sets.append((start, prev_idx))
        start = prev_idx = seq

if not sets:
    sets = [(start, prev_idx)]

print sets
