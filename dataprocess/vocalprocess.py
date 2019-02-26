"""
This code organizes the dataset suitable for VocalData.
"""


import os
from os.path import join
import shutil

DATA_DIR = '/media/skaws2003/HDD/datasets/VOCAL_mfcc'
ON_DIR = '/media/skaws2003/HDD/datasets/VOCALS/on'
OFF_DIR = '/media/skaws2003/HDD/datasets/VOCALS/off'
UNK_DIR = '/media/skaws2003/HDD/datasets/VOCALS/unk'

files = os.listdir(DATA_DIR)

if not os.path.isdir(ON_DIR):
    os.mkdir(ON_DIR)
if not os.path.isdir(OFF_DIR):
    os.mkdir(OFF_DIR)

for i,file in enumerate(files):
    if '[on vocal]' in file != -1:
        shutil.move(join(DATA_DIR,file),join(ON_DIR,file))
    elif '[off vocal]' in file != -1:
        shutil.move(join(DATA_DIR,file),join(OFF_DIR,file))
    else:
        shutil.move(join(DATA_DIR,file),join(UNK_DIR,file))

    if i % 100 == 0:
        print("%d/%d"%(i,len(files)))
