"""
This file measures the length of the dataset.
shows the max, min length of the vocals.
Current result:
 max 103384
 min 13936
"""
import numpy as np
import os


DATA_DIR = '/media/skaws2003/HDD/datasets/VOCALS'

minima = 11111110
maxima = 0
for clss in os.listdir(DATA_DIR):
    files = os.listdir(os.path.join(DATA_DIR,clss))
    for i,file in enumerate(files):
        features = np.load(os.path.join(DATA_DIR,clss,file))
        flen = features.shape[0]
        maxima = max([maxima,flen])
        minima = min([minima,flen])
        del features
        if i%100 == 0:
            print("doing '%s': (%d,%d)"%(clss,i,len(files)))

print(maxima)
print(minima)